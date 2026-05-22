#!/usr/bin/env python
"""Create a new dataset truncated to end before the second gripper-open segment.

For each episode in the source dataset, finds the frame where the gripper
crosses the threshold for the SECOND time and truncates the episode there
(exclusive). Episodes with fewer than 2 segments are dropped.

Usage:
    uv run python scripts/truncate_at_second_grip.py
    uv run python scripts/truncate_at_second_grip.py \\
        --src gaspardthrl/walleed_teleop_gaspard_clean \\
        --dst gaspardthrl/walleed_teleop_gaspard_truncated \\
        --threshold 5.0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi


def find_cutoff(gripper_series: np.ndarray, threshold: float) -> int | None:
    """Return frame index of the start of the second 'gripper > threshold' segment,
    or None if there are fewer than 2 segments."""
    above = gripper_series > threshold
    if len(above) == 0:
        return None
    # Find rising edges (False→True transitions)
    edges = []
    if above[0]:
        edges.append(0)
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            edges.append(i)
    if len(edges) < 2:
        return None
    return edges[1]  # frame index where second segment starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="gaspardthrl/walleed_teleop_gaspard_clean")
    parser.add_argument("--dst", default="gaspardthrl/walleed_teleop_gaspard_truncated")
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--gripper-dim", type=int, default=5)
    parser.add_argument("--root", default="/tmp/walleed_truncated")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading source dataset {args.src}...")
    src = LeRobotDataset(args.src, root=f"/tmp/{args.src.replace('/', '__')}")
    print(f"  {len(src.meta.episodes)} episodes, {src.meta.fps} fps")

    # Compute task lookup
    tasks_df = src.meta.tasks
    print(f"  Tasks: {tasks_df.to_dict()}")

    # Create destination dataset
    dst_root = Path(args.root)
    if dst_root.exists():
        import shutil
        shutil.rmtree(dst_root)
    print(f"\nCreating destination at {dst_root}...")
    dst = LeRobotDataset.create(
        repo_id=args.dst,
        fps=src.meta.fps,
        features=src.features,
        root=str(dst_root),
        use_videos=True,
    )

    n_kept, n_skipped, total_frames = 0, 0, 0
    n_episodes = len(src.meta.episodes)
    for i in range(n_episodes):
        ep_meta = src.meta.episodes[i]
        ep_idx = int(ep_meta["episode_index"])
        ds_from = int(ep_meta["dataset_from_index"])
        ds_to = int(ep_meta["dataset_to_index"])
        ep_length = ds_to - ds_from

        # Extract gripper series for the episode using direct parquet (faster)
        # Frames are accessed via dataset[ds_from], dataset[ds_from + 1], ...
        # But that's slow. Better: load gripper via parquet.
        # Let's just iterate through indices and look at observation.state.
        gripper = np.array([
            float(src[ds_from + i]["observation.state"][args.gripper_dim])
            for i in range(ep_length)
        ])
        cutoff = find_cutoff(gripper, args.threshold)

        if cutoff is None:
            n_skipped += 1
            print(f"  ep {ep_idx}: skipped (< 2 gripper segments)")
            continue

        # Get task string for this episode (most frames share the same task_index)
        task_idx = int(src[ds_from]["task_index"])
        task_str = tasks_df.index[tasks_df["task_index"] == task_idx][0]

        # Auto-managed columns are added by the writer; skip them here.
        AUTO_KEYS = {"index", "frame_index", "timestamp", "episode_index", "task_index"}

        # Copy frames 0..cutoff-1
        for i in range(cutoff):
            frame_data = src[ds_from + i]
            new_frame = {"task": task_str}
            for key in src.features:
                if key in AUTO_KEYS or key not in frame_data:
                    continue
                v = frame_data[key]
                if isinstance(v, torch.Tensor):
                    v = v.numpy()
                # Transpose images (C,H,W) → (H,W,C) for the writer
                if v.ndim == 3 and v.shape[0] in (1, 3):
                    v = np.transpose(v, (1, 2, 0))
                new_frame[key] = v
            dst.add_frame(new_frame)
        dst.save_episode()
        n_kept += 1
        total_frames += cutoff
        print(f"  ep {ep_idx}: kept {cutoff}/{ep_length} frames")

    print(f"\nKept {n_kept} episodes, skipped {n_skipped}, total frames: {total_frames}")

    dst.finalize()
    print("Finalized. Pushing to hub...")
    dst.push_to_hub()

    # Add v3.0 tag
    HfApi().create_tag(args.dst, tag="v3.0", repo_type="dataset", exist_ok=True)
    print(f"Done: https://huggingface.co/datasets/{args.dst}")


if __name__ == "__main__":
    main()
