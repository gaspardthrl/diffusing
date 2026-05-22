#!/usr/bin/env python
"""Phase 1+2 of A5: precompute absolute EEF poses and chunk-wise delta stats.

Outputs:
  <root>/meta/eef_poses.npy     — shape (N_total, 12): [pos(3), rotmat_flat(9)]
                                   indexed by GLOBAL dataset row index (same as batch["index"])
  <root>/meta/eef_stats.json    — MEAN_STD stats over chunk-wise EEF deltas

Run once before training A5:
  DATASET_ROOT=./data_combined uv run python scripts/precompute_eef_sidecars.py
  DATASET_ROOT=./data_combined uv run python scripts/precompute_eef_sidecars.py --no-upload
"""

import argparse
import json
import logging
import os
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

URDF_PATH = REPO_ROOT / "so101_new_calib.urdf"
URDF_SOURCES = [
    ("TheRobotStudio/SO-ARM100",
     "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf"),
    ("HuggingFace haixuantao/dora-bambot",
     "https://huggingface.co/haixuantao/dora-bambot/resolve/main/URDF/so101.urdf"),
]

EEF_FRAME_ID = 15      # gripper_frame_link  (verified in compute_eef_fk.py)
DEG2RAD = np.pi / 180  # dataset stores degrees


def _ensure_urdf() -> Path:
    if URDF_PATH.exists():
        logging.info(f"URDF: using cached {URDF_PATH}")
        return URDF_PATH
    for name, url in URDF_SOURCES:
        try:
            logging.info(f"URDF: downloading from {name} ...")
            urllib.request.urlretrieve(url, URDF_PATH)
            logging.info(f"URDF: saved to {URDF_PATH}")
            return URDF_PATH
        except Exception as e:
            logging.warning(f"URDF: {name} failed ({e}), trying next")
    raise RuntimeError("All URDF sources failed. Place so101_new_calib.urdf in repo root.")


def _load_all_rows(dataset_root: Path) -> pa.Table:
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset_root}/data")
    logging.info(f"Loading {len(parquet_files)} parquet files ...")
    table = pa.concat_tables([pq.read_table(f) for f in parquet_files])
    logging.info(f"Total rows: {len(table)}")
    return table


def _build_eef_poses(table: pa.Table) -> np.ndarray:
    """Run FK for every row. Returns float32 array (N, 12): [pos(3), rotmat_flat(9)].

    Rows are sorted by global `index` so array[i] corresponds to global dataset row i.
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(_ensure_urdf()))
    data = model.createData()

    df = table.to_pandas()
    df = df.sort_values("index").reset_index(drop=True)

    n = len(df)
    poses = np.zeros((n, 12), dtype=np.float32)

    states = np.array(df["observation.state"].tolist(), dtype=np.float64)

    logging.info(f"Running FK for {n} rows ...")
    for i in range(n):
        q = np.zeros(model.nq)
        q[:5] = states[i, :5] * DEG2RAD

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        oMf = data.oMf[EEF_FRAME_ID]
        poses[i, :3] = oMf.translation
        poses[i, 3:] = oMf.rotation.flatten()  # row-major (3,3) → (9,)

        if i % 50_000 == 0:
            logging.info(f"  FK progress: {i}/{n}")

    nan_count = np.isnan(poses).sum()
    if nan_count > 0:
        raise RuntimeError(f"FK produced {nan_count} NaN values — check URDF/joint convention")

    logging.info(f"FK complete. pos range: {poses[:,:3].min(0)} – {poses[:,:3].max(0)}")
    return poses


def _compute_stats(poses: np.ndarray, table: pa.Table, chunk_size: int) -> dict:
    """Iterate over all valid within-episode chunks, accumulate delta stats.

    delta_pos:     pos[t+k] - pos[t]               chunk-wise, shape (32, 3)
    delta_rot_6d:  6D(R[t].T @ R[t+k])             chunk-wise, shape (32, 6)
    gripper:       absolute gripper state[t+k, 5]   shape (32,)
    """
    df = table.to_pandas().sort_values("index").reset_index(drop=True)
    states = np.array(df["observation.state"].tolist(), dtype=np.float32)
    episode_indices = np.array(df["episode_index"].tolist())
    global_indices = np.array(df["index"].tolist())  # monotone 0..N-1

    # Collect valid chunk starts: within single episode, enough room for chunk_size frames
    episodes = np.unique(episode_indices)
    chunk_starts = []
    for ep in episodes:
        ep_mask = episode_indices == ep
        ep_globals = global_indices[ep_mask]
        ep_globals.sort()
        for start_global in ep_globals[: len(ep_globals) - chunk_size + 1]:
            chunk_starts.append(start_global)
    chunk_starts = np.array(chunk_starts)
    logging.info(f"Computing chunk delta stats over {len(chunk_starts)} chunks (chunk_size={chunk_size}) ...")

    pos_deltas_all = []
    rot_deltas_all = []
    gripper_all = []

    for start in chunk_starts:
        chunk_poses = poses[start : start + chunk_size]  # (32, 12)

        pos = chunk_poses[:, :3]             # (32, 3)
        R_flat = chunk_poses[:, 3:]          # (32, 9)

        pos_delta = pos - pos[0]             # (32, 3)

        # Rotation deltas: R_delta[k] = R[0].T @ R[k]
        R_ref = R_flat[0].reshape(3, 3)      # (3, 3)
        R_ref_T = R_ref.T                    # (3, 3)

        rot_6d = np.zeros((chunk_size, 6), dtype=np.float32)
        for k in range(chunk_size):
            R_k = R_flat[k].reshape(3, 3)
            R_delta = R_ref_T @ R_k          # (3, 3)
            # 6D = [col0, col1] of R_delta
            rot_6d[k] = np.concatenate([R_delta[:, 0], R_delta[:, 1]])

        gripper = states[start : start + chunk_size, 5]  # (32,) absolute

        pos_deltas_all.append(pos_delta)
        rot_deltas_all.append(rot_6d)
        gripper_all.append(gripper)

    pos_deltas_all = np.array(pos_deltas_all)   # (N_chunks, 32, 3)
    rot_deltas_all = np.array(rot_deltas_all)   # (N_chunks, 32, 6)
    gripper_all = np.array(gripper_all)         # (N_chunks, 32)

    pos_mean = pos_deltas_all.mean(axis=(0, 1)).tolist()  # (3,)
    pos_std  = pos_deltas_all.std(axis=(0, 1)).tolist()
    rot_mean = rot_deltas_all.mean(axis=(0, 1)).tolist()  # (6,)
    rot_std  = rot_deltas_all.std(axis=(0, 1)).tolist()
    gmin = float(gripper_all.min())
    gmax = float(gripper_all.max())

    # Verify std is in reasonable range
    all_stds = pos_std + rot_std
    for i, s in enumerate(all_stds):
        if s < 1e-6:
            logging.warning(f"WARN: near-zero std at dim {i}: {s:.6f} — axis may have no variation")

    logging.info(f"pos delta mean:  {[f'{v:.4f}' for v in pos_mean]}")
    logging.info(f"pos delta std:   {[f'{v:.4f}' for v in pos_std]}")
    logging.info(f"rot 6D delta mean (first 6): {[f'{v:.4f}' for v in rot_mean]}")
    logging.info(f"rot 6D delta std  (first 6): {[f'{v:.4f}' for v in rot_std]}")
    logging.info(f"gripper: min={gmin:.4f}, max={gmax:.4f}")

    # Normalized-std check (should be ~1.0 after mean/std normalization)
    logging.info("Normalized std check (all should be ~1.0 after MEAN_STD norm):")
    logging.info(f"  pos_std:          {pos_std}")
    logging.info(f"  rot_6D_std:       {rot_std}")

    return {
        "eef_pos_delta": {"mean": pos_mean, "std": pos_std},
        "eef_rot_delta": {"mean": rot_mean, "std": rot_std},
        "gripper": {"min": gmin, "max": gmax},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="gaspardthrl/walleed_fold_combined")
    parser.add_argument("--root", default=os.environ.get("DATASET_ROOT", "./data_combined"))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    poses_path = meta_dir / "eef_poses.npy"
    stats_path = meta_dir / "eef_stats.json"

    # ── Phase 1: FK for all rows ──────────────────────────────────────────────
    table = _load_all_rows(root)

    if poses_path.exists():
        logging.info(f"Phase 1: loading cached {poses_path}")
        poses = np.load(poses_path)
        assert len(poses) == len(table), (
            f"Cached poses length {len(poses)} != table length {len(table)} — delete and rerun"
        )
    else:
        poses = _build_eef_poses(table)
        np.save(poses_path, poses)
        logging.info(f"Phase 1: saved {poses_path}  shape={poses.shape}  size={poses.nbytes/1e6:.1f}MB")

    # ── Phase 2: compute chunk delta stats ────────────────────────────────────
    stats = _compute_stats(poses, table, args.chunk_size)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Phase 2: saved {stats_path}")

    print("\n=== eef_stats.json ===")
    print(json.dumps(stats, indent=2))

    # ── Upload to HF ──────────────────────────────────────────────────────────
    if not args.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for local_path, hf_name in [(poses_path, "meta/eef_poses.npy"), (stats_path, "meta/eef_stats.json")]:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hf_name,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Add {hf_name} (A5 EEF poses/stats, chunk_size={args.chunk_size})",
            )
            logging.info(f"Uploaded hf://datasets/{args.repo_id}/{hf_name}")


if __name__ == "__main__":
    main()
