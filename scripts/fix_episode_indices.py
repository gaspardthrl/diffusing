#!/usr/bin/env python
"""Recompute episode `length`, `dataset_from_index`, `dataset_to_index` from
actual parquet row counts.

The recording pipeline left wrong `length` values for some episodes (mostly 0-8),
which broke `dataset_from_index`/`dataset_to_index` (they were computed cumulatively
from those wrong lengths). The sampler then tries to access global indices that
exceed the actual data row count.

This script:
  1. Scans all data parquets to get actual row count per episode.
  2. Recomputes length + cumulative dataset_from/to_index.
  3. Updates all episodes parquets and uploads to HF.

Usage:
    DATASET_ROOT=./data_hg uv run python scripts/fix_episode_indices.py
    DATASET_ROOT=./data_hg uv run python scripts/fix_episode_indices.py --no-upload
"""

import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="gaspardthrl/walleed_hg_double_fold")
    parser.add_argument("--root", default=os.environ.get("DATASET_ROOT", "./data_hg"))
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)

    # Download dataset if needed
    if not (root / "data").exists():
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        logging.info(f"Downloading {args.repo_id} to {root} ...")
        LeRobotDataset(repo_id=args.repo_id, root=str(root))

    import pandas as pd

    # ── 1. Count actual rows per episode from data parquets ────────────
    data_dir = root / "data" / "chunk-000"
    data_parquets = sorted(data_dir.glob("file-*.parquet"))
    logging.info(f"Scanning {len(data_parquets)} data parquets...")

    actual_lengths: dict[int, int] = defaultdict(int)
    for p in data_parquets:
        df = pd.read_parquet(p, columns=["episode_index"])
        for ep_idx, count in df["episode_index"].value_counts().items():
            actual_lengths[int(ep_idx)] += int(count)

    logging.info(f"Found {len(actual_lengths)} unique episodes in data")
    total_frames = sum(actual_lengths.values())
    logging.info(f"Total frames: {total_frames}")

    # ── 2. Compute cumulative dataset_from/to_index ────────────────────
    ep_indices = sorted(actual_lengths.keys())
    cumulative = 0
    new_meta: dict[int, dict] = {}
    for ep_idx in ep_indices:
        length = actual_lengths[ep_idx]
        new_meta[ep_idx] = {
            "length": length,
            "dataset_from_index": cumulative,
            "dataset_to_index": cumulative + length,
        }
        cumulative += length

    # ── 3. Update episodes parquets ────────────────────────────────────
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    ep_parquets = sorted(episodes_dir.glob("file-*.parquet"))
    logging.info(f"Updating {len(ep_parquets)} episode parquet files...")

    n_changed = 0
    for p in ep_parquets:
        if " " in p.name:
            logging.info(f"  Skipping duplicate {p.name}")
            continue
        df = pd.read_parquet(p)
        modified = False
        for i, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            if ep_idx not in new_meta:
                continue
            new = new_meta[ep_idx]
            for col in ["length", "dataset_from_index", "dataset_to_index"]:
                if col in df.columns and int(row[col]) != new[col]:
                    df.at[i, col] = new[col]
                    modified = True
                    n_changed += 1
        if modified:
            df.to_parquet(p, index=False)
            logging.info(f"  Updated {p.name}")

    logging.info(f"Total field updates: {n_changed}")

    # ── 4. Update info.json ────────────────────────────────────────────
    import json
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_frames"] = total_frames
    info["total_episodes"] = len(actual_lengths)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logging.info(f"Updated info.json: total_episodes={len(actual_lengths)}, total_frames={total_frames}")

    # ── 5. Upload to HF ────────────────────────────────────────────────
    if not args.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for p in ep_parquets:
            if " " in p.name:
                continue
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=str(p.relative_to(root)),
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Fix episode length + dataset_from/to_index from actual data",
            )
        api.upload_file(
            path_or_fileobj=str(info_path),
            path_in_repo="meta/info.json",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Update info.json with correct total_frames and total_episodes",
        )
        logging.info(f"Uploaded to {args.repo_id}")


if __name__ == "__main__":
    main()
