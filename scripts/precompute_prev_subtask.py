#!/usr/bin/env python3
"""Precompute the per-frame `prev_subtask` lookup array for diffusion-policy training.

Reads dense subtask annotations from `<dataset_root>/meta/episodes/**/*.parquet` and
produces `<dataset_root>/meta/prev_subtask.npy`, an int8 array of length N_total_frames
indexed by the global frame index ("index" column of the data parquets).

The shift and episode-boundary handling are baked into the array:
    prev_subtask[global_idx] = labels_of_episode(ep)[frame_in_ep - 1]    if frame_in_ep > 0
                             = labels_of_episode(ep)[0]                  if frame_in_ep == 0

This matches the rule used by `scripts/train_subtask_classifier.py` (`labels[t-1] if t>0
else labels[0]`), so the diffusion policy can do a plain `arr[batch["index"]]` lookup.

Usage:
    uv run python scripts/precompute_prev_subtask.py \
        --dataset-root ~/.cache/huggingface/lerobot/hub/datasets--user--name/snapshots/<hash>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SUBTASK_NAMES = [
    "reach the towel first fold",
    "pick the towel first fold",
    "perform first fold",
    "release towel first fold",
    "reach the towel second fold",
    "pick the towel second fold",
    "perform second fold",
    "release towel second fold",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <dataset_root>/meta/prev_subtask.npy)",
    )
    args = p.parse_args()

    root: Path = args.dataset_root
    out: Path = args.output or (root / "meta" / "prev_subtask.npy")

    # 1) Read dense annotations + dataset_from_index per episode.
    ep_labels: dict[int, np.ndarray] = {}
    ep_from_index: dict[int, int] = {}
    for f in sorted(root.glob("meta/episodes/**/*.parquet")):
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            ep_from_index[ep] = int(row["dataset_from_index"])
            names = row.get("dense_subtask_names")
            if not isinstance(names, (list, np.ndarray)):
                continue
            starts = [int(x) for x in row["dense_subtask_start_frames"]]
            ends = [int(x) for x in row["dense_subtask_end_frames"]]
            n = max(ends[-1] + 1, int(row.get("length") or 0))
            labels = np.zeros(n, dtype=np.int8)
            for name, s, e in zip(names, starts, ends):
                try:
                    labels[s : e + 1] = SUBTASK_NAMES.index(name)
                except ValueError:
                    pass
            ep_labels[ep] = labels

    if not ep_labels:
        raise RuntimeError(f"No dense_subtask_names found under {root}/meta/episodes/")

    # 2) Total frame count: prefer meta/info.json (small file, always present);
    #    fall back to scanning data parquets if needed.
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        import json
        with open(info_path) as f:
            info = json.load(f)
        total = int(info["total_frames"])
    else:
        max_idx = 0
        for f in sorted(root.glob("data/**/*.parquet")):
            df = pd.read_parquet(f, columns=["index"])
            max_idx = max(max_idx, int(df["index"].max()))
        total = max_idx + 1

    # 3) Build prev_subtask: shift labels by one within each episode, clamp at boundary.
    arr = np.zeros(total, dtype=np.int8)
    for ep, labels in ep_labels.items():
        start = ep_from_index[ep]
        end = min(start + len(labels), total)
        # prev_subtask for frame_in_ep == 0 is labels[0] (matches classifier training rule).
        # prev_subtask for frame_in_ep > 0  is labels[frame_in_ep - 1].
        shifted = np.empty_like(labels)
        shifted[0] = labels[0]
        shifted[1:] = labels[:-1]
        arr[start:end] = shifted[: end - start]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print(f"Saved {arr.shape[0]} labels to {out}  (unique classes: {np.unique(arr).tolist()})")


if __name__ == "__main__":
    main()
