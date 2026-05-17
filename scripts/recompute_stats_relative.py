#!/usr/bin/env python
"""Compute relative-action stats and save as relative_stats.json (does NOT touch stats.json).

Uploads the result to the HuggingFace dataset repo so it lives alongside the dataset.
At training time, pass --dataset.stats_override_path=./data/meta/relative_stats.json
(or wherever you downloaded it) to have make_dataset patch the action stats in-place.

Usage:
    # Compute, save locally, and upload to HF:
    DATASET_ROOT=./data uv run python scripts/recompute_stats_relative.py

    # Compute only (no upload):
    DATASET_ROOT=./data uv run python scripts/recompute_stats_relative.py --no-upload
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="gaspardthrl/walleed_hg_double_fold")
    parser.add_argument(
        "--revision",
        default=None,
        help="Pin to a specific HF commit (e.g. 09a23364a3a7). Default uses latest.",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("DATASET_ROOT", "./data"),
        help="Local dataset root (must contain meta/info.json and data/ parquets).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Must match policy.horizon in the training script.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-name",
        default="relative_stats.json",
        help="Filename written to <root>/meta/. Also uploaded to the HF dataset repo.",
    )
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload.")
    parser.add_argument(
        "--gripper-dim",
        type=int,
        default=5,
        help="Action dimension index for gripper (excluded from relative actions, stays binary 0/1). "
             "Its mean/std are patched to be MIN_MAX-equivalent so MEAN_STD normalization works correctly.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_path = root / "meta" / args.output_name

    # ── 1. Load dataset ───────────────────────────────────────────────
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logging.info(f"Loading dataset {args.repo_id} from {root} (revision={args.revision or 'latest'}) ...")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=str(root), revision=args.revision)

    # ── 2. Compute relative action stats ─────────────────────────────
    from lerobot.datasets.compute_stats import compute_relative_action_stats

    logging.info("Computing relative action stats (this may take a minute) ...")
    rel_stats = compute_relative_action_stats(
        hf_dataset=dataset.hf_dataset,
        features=dataset.meta.features,
        chunk_size=args.chunk_size,
        exclude_joints=["gripper"],
        num_workers=args.num_workers,
    )

    # ── 3. Patch gripper dim mean/std to be MIN_MAX-equivalent ────────
    # Gripper is binary {0, 1} and excluded from relative actions.
    # With MEAN_STD normalization, its actual mean/std depend on dataset balance
    # and can place values outside [-1, 1], conflicting with clip_sample_range.
    # Patching to (min+max)/2, (max-min)/2 maps {min→-1, max→+1} regardless of
    # the observed distribution — identical to MIN_MAX behavior.
    g = args.gripper_dim
    g_min, g_max = rel_stats["min"][g], rel_stats["max"][g]
    rel_stats["mean"][g] = (g_min + g_max) / 2.0
    rel_stats["std"][g] = (g_max - g_min) / 2.0
    logging.info(
        f"Gripper dim {g}: patched mean={rel_stats['mean'][g]:.4f}, std={rel_stats['std'][g]:.4f} "
        f"(MIN_MAX-equivalent from min={g_min:.4f}, max={g_max:.4f})"
    )

    # ── 4. Save locally ───────────────────────────────────────────────
    # rel_stats values are numpy arrays; convert to plain lists for JSON.
    serializable = {
        "action": {k: v.tolist() for k, v in rel_stats.items()}
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logging.info(f"Saved relative stats to {out_path}")

    # ── 5. Sanity-check printout ──────────────────────────────────────
    action = serializable["action"]
    print("\nRelative action stats (arm joints are deltas, gripper is absolute):")
    print(f"  min:  {action.get('min')}")
    print(f"  max:  {action.get('max')}")
    print(f"  mean: {action.get('mean')}")

    # ── 6. Upload to HF ───────────────────────────────────────────────
    if not args.no_upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=f"meta/{args.output_name}",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add {args.output_name} (relative action stats, chunk_size={args.chunk_size})",
        )
        logging.info(f"Uploaded to hf://datasets/{args.repo_id}/meta/{args.output_name}")
        print(
            f"\nTo use at training time, add to your training script:\n"
            f"  --dataset.stats_override_path={out_path}"
        )


if __name__ == "__main__":
    main()
