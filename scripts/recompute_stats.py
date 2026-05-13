"""Recompute meta/stats.json for a LeRobot v3 dataset.

Tabular features (action, observation.state, timestamp, frame_index, episode_index,
index, task_index) are fully recomputed from raw parquet rows — min/max/mean/std/count
and quantiles q01/q10/q50/q90/q99.

Image features (observation.images.*) get min/max/mean/std/count aggregated from the
per-episode stats in meta/episodes/*.parquet (cheap, no video decoding). Image quantiles
are kept from the existing stats.json — recomputing per-pixel quantiles would require
streaming all video frames and isn't typically used at training time (LeRobot defaults
to ImageNet stats for image normalization).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TABULAR_COLS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
VECTOR_COLS = ("action", "observation.state")
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
QUANTILE_NAMES = ("q01", "q10", "q50", "q90", "q99")


def _stats_scalar(arr: np.ndarray) -> dict:
    arr = np.asarray(arr).reshape(-1)
    qs = np.quantile(arr, QUANTILES)
    out = {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.size)],
    }
    for name, q in zip(QUANTILE_NAMES, qs):
        out[name] = [float(q)]
    return out


def _stats_vector(mat: np.ndarray) -> dict:
    """mat: (N, D). Returns dict of length-D lists."""
    mat = np.asarray(mat)
    qs = np.quantile(mat, QUANTILES, axis=0)  # (5, D)
    out = {
        "min": mat.min(axis=0).astype(float).tolist(),
        "max": mat.max(axis=0).astype(float).tolist(),
        "mean": mat.mean(axis=0).astype(float).tolist(),
        "std": mat.std(axis=0).astype(float).tolist(),
        "count": [int(mat.shape[0])] * mat.shape[1],
    }
    for name, row in zip(QUANTILE_NAMES, qs):
        out[name] = row.astype(float).tolist()
    return out


def _aggregate_image_from_episodes(meta_df: pd.DataFrame, key: str) -> dict:
    """Aggregate min/max/mean/std/count for an image feature from per-episode stats.

    Per-episode columns:
      stats/{key}/min, max:   shape (C, H, W)   -> global min/max = elementwise min/max
      stats/{key}/mean, std:  shape (C, H, W)   -> combined via count-weighted formula
      stats/{key}/count:      [count] per episode
    """
    prefix = f"stats/{key}"
    counts = np.array([c[0] for c in meta_df[f"{prefix}/count"]], dtype=np.int64)  # (E,)
    # Per-episode image stats are deeply-nested object arrays (C, H, W) where H, W are
    # summary dims (1, 1 in standard LeRobot output). Use .tolist() to flatten through
    # the object dtype, then build a float64 array of shape (E, C, H, W).
    def _stack(col_name):
        return np.array([np.asarray(v, dtype=object).tolist() for v in meta_df[col_name]],
                        dtype=np.float64)
    means = _stack(f"{prefix}/mean")  # (E, C, H, W)
    stds = _stack(f"{prefix}/std")
    mins = _stack(f"{prefix}/min")
    maxs = _stack(f"{prefix}/max")

    n_total = int(counts.sum())
    # weighted mean — broadcast counts (E,) across the trailing dims of means (E, C, H, W) etc.
    w_shape = (counts.shape[0],) + (1,) * (means.ndim - 1)
    w = counts.reshape(w_shape).astype(np.float64)
    mean_g = (means * w).sum(axis=0) / n_total
    # combined variance: E[X^2] - E[X]^2 where E[X^2]_i = var_i + mean_i^2
    e_x2 = (stds**2 + means**2)
    var_g = (e_x2 * w).sum(axis=0) / n_total - mean_g**2
    var_g = np.clip(var_g, 0.0, None)  # numerical guard
    std_g = np.sqrt(var_g)
    min_g = mins.min(axis=0)
    max_g = maxs.max(axis=0)

    # Standard LeRobot global image-stat shape is (C, 1, 1). If the per-episode shape was
    # (C, 1), add a trailing singleton so the global shape matches.
    def _to_3d(arr):
        if arr.ndim == 2:
            return arr[..., None]
        return arr

    return {
        "min": _to_3d(min_g).astype(float).tolist(),
        "max": _to_3d(max_g).astype(float).tolist(),
        "mean": _to_3d(mean_g).astype(float).tolist(),
        "std": _to_3d(std_g).astype(float).tolist(),
        "count": [n_total],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Local dataset root")
    ap.add_argument("--out", default=None,
                    help="Output path (defaults to <dataset>/meta/stats.json — overwrites in place)")
    args = ap.parse_args()
    root = Path(args.dataset)
    out_path = Path(args.out) if args.out else root / "meta" / "stats.json"

    # Load existing stats.json so we can preserve image quantiles unchanged
    existing_path = root / "meta" / "stats.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}

    # Load merged meta (for image stat aggregation)
    meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    meta = pd.concat([pq.read_table(p).to_pandas() for p in meta_files], ignore_index=True)
    print(f"[1/3] meta: {len(meta)} episodes")

    # Read all raw parquet rows for tabular stats
    print("[2/3] reading raw rows for tabular stats ...")
    parts = sorted((root / "data").rglob("*.parquet"))
    df = pd.concat([pq.read_table(p).to_pandas() for p in parts], ignore_index=True)
    print(f"      {len(df)} rows across {len(parts)} files")

    stats: dict = {}

    for col in TABULAR_COLS:
        stats[col] = _stats_scalar(df[col].to_numpy())

    for col in VECTOR_COLS:
        mat = np.stack(df[col].to_numpy()).astype(np.float64)
        stats[col] = _stats_vector(mat)

    # Image features: enumerate from existing stats.json keys
    print("[3/3] aggregating image stats from per-episode meta ...")
    image_keys = [k for k in existing.keys() if k.startswith("observation.images.")]
    if not image_keys:
        # try to infer from meta columns
        prefixes = {c.split("/")[1] for c in meta.columns if c.startswith("stats/observation.images.")}
        image_keys = list(prefixes)
    for ikey in image_keys:
        agg = _aggregate_image_from_episodes(meta, ikey)
        # preserve quantiles from existing
        for qn in QUANTILE_NAMES:
            field = existing.get(ikey, {}).get(qn)
            if field is not None:
                agg[qn] = field
        stats[ikey] = agg
        print(f"      {ikey}: aggregated min/max/mean/std/count from {len(meta)} episodes "
              f"(kept quantiles from existing stats.json)")

    out_path.write_text(json.dumps(stats, indent=4))
    print(f"\nwrote {out_path}")
    # quick comparison
    if existing:
        for col in VECTOR_COLS:
            old_mean = existing.get(col, {}).get("mean")
            new_mean = stats[col]["mean"]
            if old_mean is not None:
                diff = max(abs(a - b) for a, b in zip(old_mean, new_mean))
                print(f"  {col}.mean: max abs change = {diff:.4f}")


if __name__ == "__main__":
    main()
