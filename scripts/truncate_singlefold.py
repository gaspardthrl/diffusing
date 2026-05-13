"""Truncate each episode of a LeRobot v3 dataset at the second gripper-close rising edge.

Rationale: the source dataset (single-fold teleop) ends each episode with a re-grip on
the folded towel corner. When combined with the double-fold dataset we want to keep
only the first-fold portion of each single-fold episode. This script cuts every episode
right at the start of the second gripper-close (gripper.pos crossing > threshold).

What it writes:
    data/chunk-NNN/file-XXX.parquet   - truncated rows, global `index` re-numbered
    meta/episodes/chunk-NNN/file-XXX.parquet - updated length/dataset_from_to_index,
                                                shrunk video to_timestamp,
                                                recomputed tabular per-episode stats
    meta/info.json                    - updated total_frames
    meta/tasks.parquet                - copied unchanged
    meta/stats.json                   - copied unchanged (image stats would need video decode)
    videos/...                        - hardlinked from source (untouched)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


GRIPPER_INDEX = 5            # last element of action/observation.state vector
THRESHOLD = 5.0
MIN_RUN_FRAMES = 3           # ignore brief blips below this run length


# ---------- detection ---------- #

def detect_rises(gripper: np.ndarray, threshold: float = THRESHOLD,
                 min_run: int = MIN_RUN_FRAMES) -> list[int]:
    """Return frame indices of rising edges where gripper stays above threshold for >=min_run frames."""
    closed = (gripper > threshold).astype(np.int8)
    diff = np.diff(np.concatenate(([0], closed)))
    raw = np.where(diff == 1)[0]
    out = []
    for r in raw:
        end = r
        while end < len(closed) and closed[end] == 1:
            end += 1
        if end - r >= min_run:
            out.append(int(r))
    return out


def cut_index_for_episode(obs_gripper: np.ndarray) -> int | None:
    rises = detect_rises(obs_gripper)
    if len(rises) >= 2:
        return rises[1]
    return None  # episode has only one (or zero) gripper-close - leave it untouched


# ---------- stats helpers ---------- #

def _stats_for_scalar(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    if arr.size == 0:
        # shouldn't happen, but guard anyway
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0],
                "q01": [0.0], "q10": [0.0], "q50": [0.0], "q90": [0.0], "q99": [0.0]}
    return {
        "min": [float(np.min(arr))],
        "max": [float(np.max(arr))],
        "mean": [float(np.mean(arr))],
        "std": [float(np.std(arr))],
        "count": [int(arr.size)],
        "q01": [float(np.quantile(arr, 0.01))],
        "q10": [float(np.quantile(arr, 0.10))],
        "q50": [float(np.quantile(arr, 0.50))],
        "q90": [float(np.quantile(arr, 0.90))],
        "q99": [float(np.quantile(arr, 0.99))],
    }


def _stats_for_vector(mat: np.ndarray) -> dict:
    """mat: shape (N, D). Returns dict of length-D lists."""
    mat = np.asarray(mat)
    return {
        "min": np.min(mat, axis=0).astype(float).tolist(),
        "max": np.max(mat, axis=0).astype(float).tolist(),
        "mean": np.mean(mat, axis=0).astype(float).tolist(),
        "std": np.std(mat, axis=0).astype(float).tolist(),
        "count": [int(mat.shape[0])] * mat.shape[1],
        "q01": np.quantile(mat, 0.01, axis=0).astype(float).tolist(),
        "q10": np.quantile(mat, 0.10, axis=0).astype(float).tolist(),
        "q50": np.quantile(mat, 0.50, axis=0).astype(float).tolist(),
        "q90": np.quantile(mat, 0.90, axis=0).astype(float).tolist(),
        "q99": np.quantile(mat, 0.99, axis=0).astype(float).tolist(),
    }


# ---------- main pipeline ---------- #

def pass1_compute_cuts(src_data_dir: Path) -> dict[int, dict]:
    """First pass: read every data parquet, return per-episode cut info.

    Detection uses the `action` channel (commanded gripper) rather than observation.state,
    because action leads the physical state by ~3 frames. Cutting on action ensures the
    last kept frame does NOT contain a re-grip command, which is what we want to exclude
    when stitching this single-fold dataset onto the double-fold dataset.

    Returns dict: orig_episode_index -> {orig_length, cut, new_length, keep,
                                          data_file_index, data_chunk_index}
    `keep=False` means the episode has no detectable second-rise and will be DROPPED.
    """
    info: dict[int, dict] = {}
    for parquet_path in sorted(src_data_dir.glob("chunk-*/file-*.parquet")):
        chunk_index = int(parquet_path.parent.name.split("-")[1])
        file_index = int(parquet_path.stem.split("-")[1])
        df = pq.read_table(parquet_path).to_pandas()
        for ep, sub in df.groupby("episode_index", sort=True):
            act_g = np.array([row[GRIPPER_INDEX] for row in sub["action"]])
            cut = cut_index_for_episode(act_g)
            orig_len = len(sub)
            keep = cut is not None
            new_len = cut if keep else 0
            info[int(ep)] = {
                "orig_length": orig_len,
                "cut": cut,
                "new_length": new_len,
                "keep": keep,
                "data_chunk_index": chunk_index,
                "data_file_index": file_index,
            }
    return info


def build_mappings(per_episode: dict[int, dict]) -> tuple[dict[int, int], dict[int, int]]:
    """Build orig_ep -> new_ep_index and orig_ep -> new_dataset_from_index for kept episodes."""
    ep_remap: dict[int, int] = {}
    ep_to_from: dict[int, int] = {}
    cursor = 0
    new_ep = 0
    for ep in sorted(per_episode.keys()):
        if not per_episode[ep]["keep"]:
            continue
        ep_remap[ep] = new_ep
        ep_to_from[ep] = cursor
        cursor += per_episode[ep]["new_length"]
        new_ep += 1
    return ep_remap, ep_to_from


def rewrite_data_files(src_data_dir: Path, dst_data_dir: Path,
                       per_episode: dict, ep_remap: dict, ep_to_from: dict) -> None:
    """Slice each kept episode's rows to its cut, remap episode_index, rebuild global `index`."""
    orig_schema = None
    for src_path in sorted(src_data_dir.glob("chunk-*/file-*.parquet")):
        rel = src_path.relative_to(src_data_dir)
        dst_path = dst_data_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(src_path)
        if orig_schema is None:
            orig_schema = table.schema
        df = table.to_pandas()
        kept_blocks = []
        for ep, sub in df.groupby("episode_index", sort=True):
            ep_int = int(ep)
            if not per_episode[ep_int]["keep"]:
                continue
            new_len = per_episode[ep_int]["new_length"]
            sub = sub.iloc[:new_len].copy()
            sub["episode_index"] = np.int64(ep_remap[ep_int])
            new_from = ep_to_from[ep_int]
            sub["index"] = np.arange(new_from, new_from + new_len, dtype=np.int64)
            kept_blocks.append(sub)
        if not kept_blocks:
            # this file held only dropped episodes - skip writing
            continue
        out_df = pd.concat(kept_blocks, ignore_index=True)
        out_table = pa.Table.from_pandas(out_df, schema=orig_schema, preserve_index=False)
        pq.write_table(out_table, dst_path)


def rewrite_episode_meta_files(src_meta_eps_dir: Path, dst_meta_eps_dir: Path,
                               src_data_dir: Path, per_episode: dict,
                               ep_remap: dict, ep_to_from: dict, fps: float) -> None:
    """Drop rows for dropped episodes, update bookkeeping for kept ones, recompute tabular stats.

    Image stats (stats/observation.images.front/*) are left unchanged - recomputing them
    would require decoding AV1 video, and the drift from dropping the trailing seconds
    is negligible for image normalization.
    """
    data_cache: dict[tuple[int, int], pd.DataFrame] = {}

    for src_path in sorted(src_meta_eps_dir.glob("chunk-*/file-*.parquet")):
        rel = src_path.relative_to(src_meta_eps_dir)
        dst_path = dst_meta_eps_dir / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        meta = pq.read_table(src_path).to_pandas()

        # filter to kept episodes only
        keep_mask = meta["episode_index"].apply(lambda e: per_episode[int(e)]["keep"])
        meta = meta[keep_mask].reset_index(drop=True)
        if len(meta) == 0:
            # entirely dropped - skip writing
            continue

        for ridx in range(len(meta)):
            orig_ep = int(meta.at[ridx, "episode_index"])
            info = per_episode[orig_ep]
            new_len = info["new_length"]
            new_from = ep_to_from[orig_ep]
            new_ep = ep_remap[orig_ep]

            meta.at[ridx, "episode_index"] = new_ep
            meta.at[ridx, "length"] = new_len
            meta.at[ridx, "dataset_from_index"] = new_from
            meta.at[ridx, "dataset_to_index"] = new_from + new_len

            from_ts = float(meta.at[ridx, "videos/observation.images.front/from_timestamp"])
            meta.at[ridx, "videos/observation.images.front/to_timestamp"] = (
                from_ts + new_len / fps
            )

            # recompute tabular stats from the truncated data rows
            ck, fk = info["data_chunk_index"], info["data_file_index"]
            key = (ck, fk)
            if key not in data_cache:
                data_cache[key] = pq.read_table(
                    src_data_dir / f"chunk-{ck:03d}" / f"file-{fk:03d}.parquet"
                ).to_pandas()
            d = data_cache[key]
            sub = d[d["episode_index"] == orig_ep].iloc[:new_len]
            action_mat = np.stack(sub["action"].values).astype(np.float64)
            obs_mat = np.stack(sub["observation.state"].values).astype(np.float64)

            for col, st in _stats_for_vector(action_mat).items():
                meta.at[ridx, f"stats/action/{col}"] = st
            for col, st in _stats_for_vector(obs_mat).items():
                meta.at[ridx, f"stats/observation.state/{col}"] = st

            new_index_arr = np.arange(new_from, new_from + new_len, dtype=np.int64)
            for col, st in _stats_for_scalar(new_index_arr).items():
                meta.at[ridx, f"stats/index/{col}"] = st

            # episode_index col uses the NEW (constant) value; others use the truncated rows
            scalar_stat_sources = [
                ("stats/timestamp", sub["timestamp"].to_numpy()),
                ("stats/frame_index", sub["frame_index"].to_numpy()),
                ("stats/episode_index", np.full(new_len, new_ep, dtype=np.int64)),
                ("stats/task_index", sub["task_index"].to_numpy()),
            ]
            for dst_prefix, value_array in scalar_stat_sources:
                for col, st in _stats_for_scalar(value_array).items():
                    meta.at[ridx, f"{dst_prefix}/{col}"] = st

        orig_schema = pq.read_schema(src_path)
        out_table = pa.Table.from_pandas(meta, schema=orig_schema, preserve_index=False)
        pq.write_table(out_table, dst_path)


def link_videos(src_videos_dir: Path, dst_videos_dir: Path,
                used_video_file_indices: set[tuple[int, int]] | None = None) -> None:
    """Hardlink each .mp4 (falls back to symlink across filesystems).

    If `used_video_file_indices` is provided (set of (chunk_index, file_index) tuples),
    only link videos that contain at least one kept episode.
    """
    for src in src_videos_dir.rglob("*.mp4"):
        rel = src.relative_to(src_videos_dir)
        if used_video_file_indices is not None:
            chunk_index = int(rel.parent.name.split("-")[1])
            file_index = int(src.stem.split("-")[1])
            if (chunk_index, file_index) not in used_video_file_indices:
                continue
        dst = dst_videos_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            os.symlink(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source dataset snapshot dir")
    ap.add_argument("--dst", required=True, help="Destination dataset dir")
    ap.add_argument("--limit-episodes", type=int, default=None,
                    help="If set, only truncate this many episodes (for dry runs)")
    ap.add_argument("--report", default=None, help="Path to write JSON cleaning report")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists():
        print(f"!! destination {dst} exists. Refusing to overwrite. Delete it first.", file=sys.stderr)
        sys.exit(1)
    dst.mkdir(parents=True)

    # 1. read source info.json
    with open(src / "meta" / "info.json") as f:
        info_json = json.load(f)
    fps = float(info_json["fps"])
    orig_total_frames = info_json["total_frames"]
    orig_total_episodes = info_json["total_episodes"]

    # 2. pass 1: compute cuts (per-episode keep/drop decision)
    print("[1/5] computing cuts per episode ...", flush=True)
    per_ep = pass1_compute_cuts(src / "data")
    n_total = len(per_ep)
    n_kept = sum(1 for v in per_ep.values() if v["keep"])
    n_dropped = n_total - n_kept
    print(f"      {n_total} episodes scanned: {n_kept} kept (truncated), {n_dropped} dropped")
    if n_dropped > 0:
        dropped_eps = [ep for ep, v in per_ep.items() if not v["keep"]]
        print(f"      dropped episode indices: {dropped_eps}")

    if args.limit_episodes is not None:
        # debug knob: drop everything past the first N kept episodes
        kept_sorted = [ep for ep in sorted(per_ep.keys()) if per_ep[ep]["keep"]]
        for ep in kept_sorted[args.limit_episodes:]:
            per_ep[ep]["keep"] = False
            per_ep[ep]["new_length"] = 0
        print(f"      (dry-run: only first {args.limit_episodes} kept episodes will be written)")

    # 3. build new episode_index remap and global from-index map
    ep_remap, ep_to_from = build_mappings(per_ep)
    new_total_episodes = len(ep_remap)
    new_total_frames = sum(per_ep[ep]["new_length"] for ep in ep_remap)
    print(f"      new total episodes: {new_total_episodes} (was {info_json['total_episodes']})")
    print(f"      new total frames:   {new_total_frames} (was {info_json['total_frames']})")

    # 4. rewrite data parquets
    print("[2/5] rewriting data parquets ...", flush=True)
    rewrite_data_files(src / "data", dst / "data", per_ep, ep_remap, ep_to_from)

    # 5. rewrite episode meta parquets
    print("[3/5] rewriting meta/episodes parquets ...", flush=True)
    rewrite_episode_meta_files(src / "meta" / "episodes", dst / "meta" / "episodes",
                               src / "data", per_ep, ep_remap, ep_to_from, fps)

    # 6. copy tasks.parquet and stats.json unchanged
    print("[4/5] copying meta/tasks.parquet + meta/stats.json ...", flush=True)
    (dst / "meta").mkdir(exist_ok=True)
    shutil.copy(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    shutil.copy(src / "meta" / "stats.json", dst / "meta" / "stats.json")

    # 7. write updated info.json (episodes + frames + splits)
    info_json["total_episodes"] = new_total_episodes
    info_json["total_frames"] = new_total_frames
    info_json["splits"] = {"train": f"0:{new_total_episodes}"}
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(info_json, f, indent=4)

    # 8. hardlink only the videos that still have a kept episode in them
    print("[5/5] linking videos ...", flush=True)
    used_video_files: set[tuple[int, int]] = set()
    # video file index is recorded per episode in meta/episodes; pull from there
    for src_meta_path in sorted((src / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        m = pq.read_table(src_meta_path).to_pandas()
        for ridx in range(len(m)):
            orig_ep = int(m.at[ridx, "episode_index"])
            if not per_ep[orig_ep]["keep"]:
                continue
            ck = int(m.at[ridx, "videos/observation.images.front/chunk_index"])
            fk = int(m.at[ridx, "videos/observation.images.front/file_index"])
            used_video_files.add((ck, fk))
    link_videos(src / "videos", dst / "videos", used_video_files)

    # 9. copy README if present
    src_readme = src / "README.md"
    if src_readme.exists():
        shutil.copy(src_readme, dst / "README.md")

    # report
    if args.report:
        cuts = {int(ep): {"orig_length": v["orig_length"], "cut": v["cut"],
                          "new_length": v["new_length"], "keep": v["keep"],
                          "new_episode_index": ep_remap.get(ep)}
                for ep, v in per_ep.items()}
        report = {
            "source": str(src),
            "destination": str(dst),
            "fps": fps,
            "gripper_threshold": THRESHOLD,
            "min_run_frames": MIN_RUN_FRAMES,
            "episodes_total": n_total,
            "episodes_kept": new_total_episodes,
            "episodes_dropped": n_total - new_total_episodes,
            "dropped_episode_indices": [ep for ep in sorted(per_ep) if not per_ep[ep]["keep"]],
            "original_total_episodes": orig_total_episodes,
            "original_total_frames": orig_total_frames,
            "new_total_frames": new_total_frames,
            "frames_removed": sum(v["orig_length"] - v["new_length"] for v in per_ep.values()),
            "per_episode": cuts,
        }
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"      wrote report → {args.report}")

    print("done.")


if __name__ == "__main__":
    main()
