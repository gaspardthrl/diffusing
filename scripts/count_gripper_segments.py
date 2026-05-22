#!/usr/bin/env python
"""Count contiguous segments where gripper pos > THRESHOLD per episode.

Loads all episodes from a LeRobot dataset, looks at the gripper dimension
(last column of observation.state), and counts how many separate "gripper
open" segments occur within each episode. Reports descriptive statistics.

Usage:
    uv run python scripts/count_gripper_segments.py
    uv run python scripts/count_gripper_segments.py --threshold 5
    uv run python scripts/count_gripper_segments.py --repo-id gaspardthrl/walleed_teleop_gaspard_clean
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from pathlib import Path


def count_segments(values: np.ndarray, threshold: float) -> int:
    """Count rising edges from <=threshold to >threshold."""
    above = values > threshold
    if len(above) == 0:
        return 0
    # A segment starts at the first frame above, or whenever we transition False→True
    transitions = (above[1:] & ~above[:-1]).sum()
    return int(transitions + (1 if above[0] else 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="gaspardthrl/walleed_teleop_gaspard_clean")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--field", default="observation.state",
                        choices=["observation.state", "action"],
                        help="Which column to read gripper from")
    parser.add_argument("--gripper-dim", type=int, default=5,
                        help="Index of gripper in the column (default 5 = last for SO-101)")
    args = parser.parse_args()

    root = Path(snapshot_download(
        args.repo_id,
        repo_type="dataset",
        local_dir=f"/tmp/{args.repo_id.replace('/', '__')}",
        allow_patterns=["data/**", "meta/info.json"],
    ))

    # Load all data parquets, accumulate gripper per episode
    print(f"Loading data from {root}...")
    grippers_per_ep = defaultdict(list)
    for p in sorted((root / "data" / "chunk-000").glob("file-*.parquet")):
        df = pd.read_parquet(p, columns=["episode_index", "frame_index", args.field])
        # action / observation.state are stored as lists/arrays per row
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            fi = int(row["frame_index"])
            value = float(row[args.field][args.gripper_dim])
            grippers_per_ep[ep].append((fi, value))

    print(f"Loaded {len(grippers_per_ep)} episodes")

    # For each episode, sort by frame_index and count segments
    segment_counts = []
    for ep in sorted(grippers_per_ep):
        frames = sorted(grippers_per_ep[ep], key=lambda x: x[0])
        values = np.array([v for _, v in frames])
        segment_counts.append(count_segments(values, args.threshold))

    arr = np.array(segment_counts)

    print(f"\n=== Gripper segments (pos > {args.threshold}) from {args.field}[{args.gripper_dim}] ===")
    print(f"Episodes:     {len(arr)}")
    print(f"Mean:         {arr.mean():.3f}")
    print(f"Std:          {arr.std():.3f}")
    print(f"Min / Max:    {arr.min()} / {arr.max()}")
    print(f"Median:       {np.median(arr):.1f}")
    print(f"Q25 / Q75:    {np.percentile(arr, 25):.1f} / {np.percentile(arr, 75):.1f}")
    print(f"\nDistribution:")
    from collections import Counter
    for n_seg, count in sorted(Counter(arr.tolist()).items()):
        bar = "█" * count
        print(f"  {n_seg:>3} segments: {count:>4}  {bar}")


if __name__ == "__main__":
    main()
