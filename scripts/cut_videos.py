"""Cut LeRobot v3 video files to match an already-truncated dataset's metadata.

Run AFTER scripts/truncate_singlefold.py. That script writes parquet/meta but leaves
videos hardlinked from the source snapshot; this script replaces those hardlinks with
re-encoded video files that contain only the kept episode segments, and rewrites the
per-episode `from_timestamp`/`to_timestamp` so they're contiguous within each new video.

Per video file:
    1. Read meta/episodes to find every episode in this video file
    2. Build keep-segments from each episode's (orig_from_ts, new_to_ts)
       NOTE: at this stage, the meta produced by truncate_singlefold.py has from_ts =
       ORIGINAL and to_ts = from_ts + new_length/fps. So the segments to extract from the
       source video are exactly those (from_ts, to_ts) pairs.
    3. Decode source video, keep only frames whose time falls inside any segment,
       re-encode as AV1 (libsvtav1, preset 12)
    4. Rewrite the meta's from_ts/to_ts so they pack contiguously starting at 0
"""
from __future__ import annotations

import argparse
import fractions
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def cut_one_video(
    src_video: Path,
    dst_video: Path,
    segments: list[tuple[float, float]],
    fps: float,
    codec: str = "libsvtav1",
    preset: str = "12",
    crf: str = "30",
) -> int:
    """Re-encode src_video keeping only frames within any (from, to) second-range.

    Returns total kept frame count.
    """
    segments = sorted(segments)
    # tolerance to absorb float jitter between metadata seconds and frame timestamps
    eps = 0.5 / fps

    container_in = av.open(str(src_video))
    stream_in = container_in.streams.video[0]
    width, height = stream_in.codec_context.width, stream_in.codec_context.height
    pix_fmt = stream_in.codec_context.pix_fmt or "yuv420p"

    dst_video.parent.mkdir(parents=True, exist_ok=True)
    container_out = av.open(str(dst_video), mode="w")
    rate_frac = fractions.Fraction(int(round(fps)), 1)
    stream_out = container_out.add_stream(codec, rate=rate_frac)
    stream_out.width = width
    stream_out.height = height
    stream_out.pix_fmt = pix_fmt
    stream_out.options = {"preset": preset, "crf": crf}

    seg_iter = iter(segments)
    cur_from, cur_to = next(seg_iter)
    done_segments = False

    out_frame_idx = 0
    for in_frame in container_in.decode(stream_in):
        if in_frame.pts is None:
            continue
        t = float(in_frame.pts * stream_in.time_base)

        # advance segment pointer past frames before current segment
        while not done_segments and t >= cur_to - eps:
            try:
                cur_from, cur_to = next(seg_iter)
            except StopIteration:
                done_segments = True
                break
        if done_segments:
            break
        if t < cur_from - eps:
            continue

        # Build a fresh VideoFrame so we control pts in the OUTPUT stream's time_base.
        # Re-encode through an ndarray round-trip — safer than reusing input frame whose
        # time_base ties it to the source stream.
        out_frame = av.VideoFrame.from_ndarray(
            in_frame.to_ndarray(format=pix_fmt), format=pix_fmt,
        )
        out_frame.pts = out_frame_idx
        out_frame_idx += 1
        for packet in stream_out.encode(out_frame):
            container_out.mux(packet)

    # flush encoder
    for packet in stream_out.encode(None):
        container_out.mux(packet)
    container_out.close()
    container_in.close()
    return out_frame_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True,
                    help="The truncated dataset dir (output of truncate_singlefold.py).")
    ap.add_argument("--src", required=True,
                    help="The original source dataset snapshot dir (videos + meta source).")
    ap.add_argument("--report", required=True,
                    help="Cleaning report JSON from truncate_singlefold.py.")
    ap.add_argument("--video-key", default="observation.images.front")
    ap.add_argument("--codec", default="libsvtav1")
    ap.add_argument("--preset", default="12")
    ap.add_argument("--crf", default="30")
    ap.add_argument("--only-file", type=int, default=None,
                    help="If set, only cut this video file_index (for testing).")
    args = ap.parse_args()

    dst = Path(args.dst)
    src = Path(args.src)
    video_key = args.video_key

    import json
    info = json.loads((dst / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    # Cleaning report: orig_episode_index (str in JSON) -> {orig_length, cut, new_length,
    #                                                       keep, new_episode_index}
    report = json.loads(Path(args.report).read_text())
    per_ep_report = {int(k): v for k, v in report["per_episode"].items()}

    # Build segments by reading the SOURCE meta (which has original timestamps for every
    # original episode_index). This makes the script idempotent — re-running won't drift
    # because we never read dst's already-rewritten timestamps.
    src_meta_files = sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    by_video: dict[tuple[int, int], list[dict]] = {}
    for meta_path in src_meta_files:
        m = pq.read_table(meta_path).to_pandas()
        for _, row in m.iterrows():
            orig_ep = int(row["episode_index"])
            rep = per_ep_report.get(orig_ep)
            if rep is None or not rep["keep"]:
                continue  # dropped or unknown — skip
            new_ep = int(rep["new_episode_index"])
            new_length = int(rep["new_length"])
            vk_chunk = int(row[f"videos/{video_key}/chunk_index"])
            vk_file = int(row[f"videos/{video_key}/file_index"])
            from_ts = float(row[f"videos/{video_key}/from_timestamp"])
            by_video.setdefault((vk_chunk, vk_file), []).append({
                "new_episode_index": new_ep,
                "orig_from_ts": from_ts,
                "length": new_length,
            })

    # Sort each video's episode list by orig_from_ts (matches concat order in source video)
    for k in by_video:
        by_video[k].sort(key=lambda r: r["orig_from_ts"])

    # Load dst meta (to be updated in place)
    dst_meta_files = sorted((dst / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    per_file_meta: dict[Path, pd.DataFrame] = {p: pq.read_table(p).to_pandas() for p in dst_meta_files}

    # index lookup: new_episode_index -> (meta_path, row_idx)
    ep_lookup: dict[int, tuple[Path, int]] = {}
    for meta_path, meta_df in per_file_meta.items():
        for ridx, ep in enumerate(meta_df["episode_index"]):
            ep_lookup[int(ep)] = (meta_path, ridx)

    # Cut each video
    total_files = len(by_video)
    for i, ((ck, fk), eps) in enumerate(sorted(by_video.items()), 1):
        if args.only_file is not None and fk != args.only_file:
            continue
        rel = Path(f"videos/{video_key}/chunk-{ck:03d}/file-{fk:03d}.mp4")
        src_video = src / rel
        dst_video = dst / rel

        segments = [(e["orig_from_ts"], e["orig_from_ts"] + e["length"] / fps) for e in eps]
        total_keep_frames = sum(e["length"] for e in eps)
        total_keep_seconds = sum(t - f for f, t in segments)

        # remove existing hardlink/file so we can write fresh
        if dst_video.exists() or dst_video.is_symlink():
            dst_video.unlink()

        print(f"[{i}/{total_files}] cutting {rel} : {len(eps)} eps, "
              f"{total_keep_frames} frames ({total_keep_seconds:.1f}s) ...", flush=True)
        t0 = time.time()
        n_written = cut_one_video(src_video, dst_video, segments, fps,
                                  codec=args.codec, preset=args.preset, crf=args.crf)
        dt = time.time() - t0
        print(f"           wrote {n_written} frames in {dt:.1f}s "
              f"({n_written/dt:.0f} fps)", flush=True)
        if n_written != total_keep_frames:
            print(f"  !! frame count mismatch: expected {total_keep_frames}, got {n_written}",
                  file=sys.stderr)

        # rewrite meta: from_ts/to_ts become cumulative in the new video
        cursor = 0.0
        for e in eps:
            new_from = cursor
            new_to = cursor + e["length"] / fps
            meta_path, row_idx = ep_lookup[e["new_episode_index"]]
            m = per_file_meta[meta_path]
            m.at[row_idx, f"videos/{video_key}/from_timestamp"] = new_from
            m.at[row_idx, f"videos/{video_key}/to_timestamp"] = new_to
            cursor = new_to

    # Persist updated meta files
    for meta_path, meta_df in per_file_meta.items():
        schema = pq.read_schema(meta_path)
        table = pa.Table.from_pandas(meta_df, schema=schema, preserve_index=False)
        pq.write_table(table, meta_path)
    print("meta from/to timestamps updated.")


if __name__ == "__main__":
    main()
