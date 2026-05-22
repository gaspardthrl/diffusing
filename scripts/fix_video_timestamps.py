#!/usr/bin/env python
"""Fix video from_timestamp/to_timestamp for episodes in re-encoded files.

After lerobot-edit-dataset's cleanup, some video files were correctly shortened
(deleted episodes' frames removed), but the surviving episodes' from/to timestamps
were left pointing to PRE-deletion positions. This script:

  1. For each video file, sums the surviving episodes' actual durations (length/fps).
  2. If that sum exceeds the file's real duration, the timestamps are stale.
  3. Rewrites them cumulatively starting at 0 within each file.

File-001 cannot be fixed (the video itself is broken, 0 bytes); episodes pointing
there must be filtered out at training time.

Usage:
    DATASET_ROOT=./data_hg uv run python scripts/fix_video_timestamps.py
    DATASET_ROOT=./data_hg uv run python scripts/fix_video_timestamps.py --no-upload
"""

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import av
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_video_duration(path: Path) -> float:
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                return 0.0
            s = c.streams.video[0]
            return float(s.duration * s.time_base) if s.duration else 0.0
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="gaspardthrl/walleed_hg_double_fold")
    parser.add_argument("--root", default=os.environ.get("DATASET_ROOT", "./data_hg"))
    parser.add_argument("--video-key", default="observation.images.front")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    video_dir = root / "videos" / args.video_key / "chunk-000"
    episodes_dir = root / "meta" / "episodes" / "chunk-000"

    # Download if needed
    if not video_dir.exists():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        logging.info(f"Downloading {args.repo_id} to {root} ...")
        LeRobotDataset(repo_id=args.repo_id, root=str(root))

    # 1. Get video file durations
    file_durations = {}
    for vf in sorted(video_dir.glob("file-*.mp4")):
        fi = int(vf.stem.split("-")[1])
        file_durations[fi] = get_video_duration(vf)
    logging.info(f"Found {len(file_durations)} video files")

    # 2. Load all episodes parquets
    from_col = f"videos/{args.video_key}/from_timestamp"
    to_col   = f"videos/{args.video_key}/to_timestamp"
    file_col = f"videos/{args.video_key}/file_index"

    all_dfs = []
    for p in sorted(episodes_dir.glob("*.parquet")):
        if " " in p.name:
            continue
        all_dfs.append((p, pd.read_parquet(p)))

    # Collect (file_idx, episode_idx, length, original_from, original_to)
    by_file = defaultdict(list)
    for path, df in all_dfs:
        for i, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            fi = int(row[file_col])
            length = int(row["length"])
            from_ts = float(row[from_col])
            by_file[fi].append({
                "path": path, "row": i, "ep_idx": ep_idx,
                "length": length, "from_ts": from_ts,
            })

    # 3. Fix each file
    fps = args.fps
    n_fixed_eps = 0
    n_skipped_files = []
    for fi in sorted(by_file):
        eps = by_file[fi]
        # Sort by original from_ts (preserves recording order within file)
        eps.sort(key=lambda e: e["from_ts"])

        cum_dur = sum(e["length"] / fps for e in eps)
        file_dur = file_durations.get(fi, 0)

        if file_dur <= 0:
            logging.warning(f"  file-{fi:03d}: video duration = {file_dur:.2f}s (broken), skipping")
            n_skipped_files.append(fi)
            continue

        if abs(cum_dur - file_dur) < 0.5:
            # Either already correct, or off by < 0.5s — leave alone
            logging.info(f"  file-{fi:03d}: episodes sum={cum_dur:.2f}s, file={file_dur:.2f}s → OK")
            continue

        logging.info(f"  file-{fi:03d}: episodes sum={cum_dur:.2f}s, file={file_dur:.2f}s → fixing")
        cursor = 0.0
        for ep in eps:
            new_from = cursor
            new_to = cursor + ep["length"] / fps
            ep["new_from"] = new_from
            ep["new_to"] = new_to
            cursor = new_to
            n_fixed_eps += 1

    # 4. Apply corrections
    corrections = {}
    for fi_eps in by_file.values():
        for ep in fi_eps:
            if "new_from" in ep:
                corrections[(ep["path"], ep["row"])] = ep

    logging.info(f"\nFixing {n_fixed_eps} episodes across {len(set(k[0] for k in corrections))} files")

    for path, df in all_dfs:
        modified = False
        for i in range(len(df)):
            key = (path, i)
            if key in corrections:
                ep = corrections[key]
                df.at[i, from_col] = round(ep["new_from"], 6)
                df.at[i, to_col] = round(ep["new_to"], 6)
                modified = True
        if modified:
            df.to_parquet(path, index=False)

    if n_skipped_files:
        logging.info(f"\n⚠️  Skipped broken video files: {n_skipped_files}")
        logging.info(f"   Episodes in these files cannot be fixed and must be filtered at training time.")

    # 5. Upload
    if not args.no_upload and corrections:
        from huggingface_hub import HfApi
        api = HfApi()
        for path, _ in all_dfs:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=str(path.relative_to(root)),
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Fix from/to_timestamp for survivors in re-encoded video files",
            )
        logging.info(f"Uploaded corrected metadata to {args.repo_id}")


if __name__ == "__main__":
    main()
