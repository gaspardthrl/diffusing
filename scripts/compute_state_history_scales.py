#!/usr/bin/env python
"""Compute per-(motor, lag) std of (q[t-lag] - q[t]) for tanh-scaling state-history features.

Outputs a JSON file, prints the CLI flag string to paste into the training script,
and uploads the result to the HuggingFace dataset repo.

Usage:
    uv run python scripts/compute_state_history_scales.py \\
        --repo-id gaspardthrl/walleed_fold_combined \\
        --motors 0,1 --lags 5,10

    # Skip upload:
    uv run python scripts/compute_state_history_scales.py --no-upload

Output:
    state_history_scales.json saved locally and uploaded to meta/state_history_scales.json on HF
    Prints: --policy.state_history_scales='[s0,s1,...]'
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="gaspardthrl/walleed_fold_combined")
    parser.add_argument("--motors", default="0,1",
                        help="Comma-separated motor indices (e.g., '0,1')")
    parser.add_argument("--lags", default="5,10",
                        help="Comma-separated lag values in frames (e.g., '5,10')")
    parser.add_argument("--output", default="state_history_scales.json")
    parser.add_argument("--no-upload", action="store_true", help="Skip HuggingFace upload.")
    args = parser.parse_args()

    motors = parse_int_list(args.motors)
    lags = parse_int_list(args.lags)
    max_lag = max(lags)

    import pandas as pd

    root = Path(snapshot_download(
        args.repo_id,
        repo_type="dataset",
        local_dir=f"/tmp/scales__{args.repo_id.replace('/', '__')}",
        allow_patterns=["data/**", "meta/info.json"],
    ))

    # Load gripper/state from each data parquet, grouped by episode
    print("Loading state per episode...")
    state_per_ep: dict[int, list] = defaultdict(list)
    for p in sorted((root / "data" / "chunk-000").glob("file-*.parquet")):
        df = pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state"])
        for _, row in df.iterrows():
            state_per_ep[int(row["episode_index"])].append((int(row["frame_index"]), row["observation.state"]))

    # For each (motor, lag), accumulate (q[t-lag] - q[t]) across all episodes
    deltas_per_feat: dict[tuple[int, int], list[float]] = defaultdict(list)
    for ep_idx, frames in state_per_ep.items():
        frames.sort(key=lambda x: x[0])
        arr = np.array([np.asarray(s, dtype=np.float32) for _, s in frames])  # [T, state_dim]
        T = len(arr)
        if T <= max_lag:
            continue
        for m in motors:
            for lag in lags:
                # q[t-lag] - q[t], for t in [lag, T-1]
                diff = arr[: T - lag, m] - arr[lag:, m]
                deltas_per_feat[(m, lag)].extend(diff.tolist())

    # Compute std per feature, in motor-major order
    scales = []
    print(f"\nPer-feature std of (q[t-lag] - q[t]):")
    print(f"{'motor':>6}  {'lag':>4}  {'n':>8}  {'std':>10}")
    for m in motors:
        for lag in lags:
            diffs = np.array(deltas_per_feat[(m, lag)], dtype=np.float64)
            s = float(diffs.std())
            scales.append(s)
            print(f"  {m:>4}    {lag:>3}  {len(diffs):>8}  {s:>9.4f}")

    # Save and print
    out = {"motors": motors, "lags": lags, "scales": scales}
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")

    flag = "[" + ",".join(f"{s:.4f}" for s in scales) + "]"
    print(f"\nPaste into train_dit_fm_absolute_temporal.sh:")
    print(f"  --policy.state_history_motors='{motors}' \\")
    print(f"  --policy.state_history_lags='{lags}' \\")
    print(f"  --policy.state_history_scales='{flag}' \\")

    if not args.no_upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=f"meta/{out_path.name}",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add {out_path.name} (state history scales, motors={motors}, lags={lags})",
        )
        print(f"\nUploaded to hf://datasets/{args.repo_id}/meta/{out_path.name}")


if __name__ == "__main__":
    main()
