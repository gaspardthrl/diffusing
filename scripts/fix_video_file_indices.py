#!/usr/bin/env python
"""Fix video file_index and from/to_timestamp in episode metadata.

Root cause: lerobot-record stored from_timestamp/to_timestamp as GLOBAL cumulative
values (sum of all previous episode durations across the whole dataset), but video
files reset timestamps to 0 at the start of each file. The decoder computes:

    seek_ts = from_timestamp + parquet_frame_ts   (dataset_reader.py:242)

So for an episode in file-001 with global from_timestamp=188s, it opens file-001
but seeks to 338s — which doesn't exist in a 438s file that starts at 0.

Fix:
  1. Get each video file's duration (PyAV container metadata, fast).
  2. Sort episodes by global from_timestamp (recording order).
  3. Greedily bin episodes into files (fill file-000, then file-001, etc.).
  4. Within each file, recompute file-local from/to timestamps as cumulative offsets.
  5. Save + upload corrected episodes parquets to HF.

Usage:
    DATASET_ROOT=./data uv run python scripts/fix_video_file_indices.py
    DATASET_ROOT=./data uv run python scripts/fix_video_file_indices.py --no-upload
    DATASET_ROOT=./data uv run python scripts/fix_video_file_indices.py \\
        --repo-id gaspardthrl/walleed_teleop_gaspard
"""

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_video_duration(video_path: Path) -> float:
    import av
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / 1_000_000)
        # Slow fallback: read all frames
        dur = 0.0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts is not None:
                    dur = max(dur, float(frame.pts * stream.time_base))
        return dur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="gaspardthrl/walleed_hg_double_fold")
    parser.add_argument("--root", default=os.environ.get("DATASET_ROOT", "./data"))
    parser.add_argument("--video-key", default="observation.images.front")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    video_dir = root / "videos" / args.video_key / "chunk-000"
    episodes_dir = root / "meta" / "episodes"

    # ── 0. Download dataset if not already local ────────────────────────
    if not video_dir.exists():
        logging.info(f"Downloading {args.repo_id} to {root} ...")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        LeRobotDataset(repo_id=args.repo_id, root=str(root))
        logging.info("Download complete.")

    # ── 1. Get duration of each video file (fast via container header) ──
    video_files = sorted(video_dir.glob("file-*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    logging.info(f"Found {len(video_files)} video files")

    file_durations: list[tuple[int, float]] = []
    for vf in video_files:
        file_idx = int(vf.stem.split("-")[1])
        dur = get_video_duration(vf)
        file_durations.append((file_idx, dur))
        logging.info(f"  file-{file_idx:03d}: {dur:.3f}s")
    file_durations.sort(key=lambda x: x[0])

    # ── 2. Load all episode metadata ────────────────────────────────────
    import pandas as pd

    from_col = f"videos/{args.video_key}/from_timestamp"
    to_col   = f"videos/{args.video_key}/to_timestamp"
    file_col = f"videos/{args.video_key}/file_index"

    ep_parquets = sorted(episodes_dir.glob("*/*.parquet"))
    all_dfs: list[tuple[Path, pd.DataFrame]] = [(p, pd.read_parquet(p)) for p in ep_parquets]

    episodes: list[dict] = []
    for path, df in all_dfs:
        for i, row in df.iterrows():
            if from_col not in row:
                continue
            episodes.append({
                "path": path,
                "row": i,
                "ep_idx": int(row["episode_index"]),
                "global_from": float(row[from_col]),
                "duration": float(row[to_col]) - float(row[from_col]),
            })

    episodes.sort(key=lambda e: e["global_from"])
    logging.info(f"Loaded {len(episodes)} episodes")

    # ── 3. Greedily bin episodes into files ─────────────────────────────
    # Each file holds episodes until its capacity (duration) is filled.
    OVERFLOW_TOLERANCE = 1.0  # allow up to 1s of overrun at file boundaries

    fi = 0  # current file index into file_durations list
    local_cursor = 0.0  # accumulated duration within current file

    for ep in episodes:
        dur = ep["duration"]
        # Advance to next file when this episode would overflow current file
        while (fi < len(file_durations) - 1 and
               local_cursor + dur > file_durations[fi][1] + OVERFLOW_TOLERANCE):
            fi += 1
            local_cursor = 0.0

        ep["new_file_idx"]  = file_durations[fi][0]
        ep["new_from"]      = local_cursor
        ep["new_to"]        = local_cursor + dur
        local_cursor       += dur

    # ── 4. Apply corrections ─────────────────────────────────────────────
    corrections: dict[tuple, dict] = {(e["path"], e["row"]): e for e in episodes}
    n_changed = 0

    for path, df in all_dfs:
        modified = False
        for i, row in df.iterrows():
            ep = corrections.get((path, i))
            if ep is None:
                continue
            old_fi = int(row[file_col]) if file_col in row else -1
            if (old_fi != ep["new_file_idx"] or
                    abs(float(row[from_col]) - ep["new_from"]) > 0.001):
                n_changed += 1
            df.at[i, file_col] = ep["new_file_idx"]
            df.at[i, from_col] = round(ep["new_from"], 6)
            df.at[i, to_col]   = round(ep["new_to"],   6)
            modified = True

        if modified:
            df.to_parquet(path, index=False)

    logging.info(f"Fixed {n_changed}/{len(episodes)} episodes")

    # Verify: print a few episodes that previously had file_index=0 but should be elsewhere
    for ep in episodes[:5] + [e for e in episodes if e["new_file_idx"] != 0][:3]:
        logging.info(
            f"  ep {ep['ep_idx']:3d}: file-{ep['new_file_idx']:03d} "
            f"from={ep['new_from']:.2f}s to={ep['new_to']:.2f}s"
        )

    # ── 5. Upload to HF ──────────────────────────────────────────────────
    if not args.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for path, _ in all_dfs:
            rel = path.relative_to(root)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=str(rel),
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Fix video file_index + from/to_timestamp (recording pipeline bug)",
            )
        logging.info(f"Uploaded corrected episodes metadata to {args.repo_id}")
    else:
        logging.info("Skipped HF upload (--no-upload)")


if __name__ == "__main__":
    main()
