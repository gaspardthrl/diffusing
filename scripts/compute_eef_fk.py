#!/usr/bin/env python
"""Compute EEF poses from SO-101 dataset episode via pinocchio FK.

Prerequisite check for experiment A5 (relative EEF delta actions).
"""
import sys
import os
import urllib.request
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent

# ── Step 1: Get URDF ──────────────────────────────────────────────────────────
URDF_PATH = REPO_ROOT / "so101.urdf"
URDF_SOURCES = [
    ("TheRobotStudio/SO-ARM100 (canonical)",
     "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf"),
    ("HuggingFace haixuantao/dora-bambot (fallback)",
     "https://huggingface.co/haixuantao/dora-bambot/resolve/main/URDF/so101.urdf"),
]

if URDF_PATH.exists():
    print(f"[URDF] Using cached: {URDF_PATH}")
else:
    for source_name, url in URDF_SOURCES:
        try:
            print(f"[URDF] Trying {source_name} ...")
            urllib.request.urlretrieve(url, URDF_PATH)
            print(f"[URDF] Downloaded from {source_name}")
            break
        except Exception as e:
            print(f"[URDF] Failed ({e}), trying next source...")
    else:
        print("[URDF] ERROR: All sources failed. Place so101.urdf in repo root manually.")
        sys.exit(1)

# ── Step 2: Load FK via pinocchio ─────────────────────────────────────────────
import pinocchio as pin

model = pin.buildModelFromUrdf(str(URDF_PATH))
data = model.createData()

print(f"\n[Pinocchio] model.nq = {model.nq}  (expected 6)")
print(f"[Pinocchio] Joint names (order must match dataset columns):")
for i in range(1, model.njoints):
    print(f"  [{i-1}] {model.names[i]}")
print(f"[Pinocchio] All frame names:")
for i in range(model.nframes):
    print(f"  [{i}] {model.frames[i].name}")

EEF_CANDIDATES = ["gripper_frame_link", "tool0", "end_effector", "gripper", "wrist_roll_link",
                  "gripper_link", "ee_link", "flange", "link6"]
frame_id = None
eef_name = None
for name in EEF_CANDIDATES:
    try:
        fid = model.getFrameId(name)
        # getFrameId returns nframes if not found (doesn't raise)
        if fid < model.nframes:
            frame_id = fid
            eef_name = name
            print(f"\n[EEF] Frame found: '{name}' (id={fid})")
            break
    except Exception:
        continue

if frame_id is None:
    # Fall back to last frame
    frame_id = model.nframes - 1
    eef_name = model.frames[frame_id].name
    print(f"\n[EEF] No candidate matched. Using last frame: '{eef_name}' (id={frame_id})")

# ── Step 3: Load episode from dataset ────────────────────────────────────────
import pyarrow.parquet as pq

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", str(REPO_ROOT / "data_combined")))
parquet_files = sorted((DATASET_ROOT / "data").rglob("*.parquet"))
if not parquet_files:
    print(f"[Dataset] ERROR: no parquet files found under {DATASET_ROOT}/data")
    sys.exit(1)

# Load all parquets and find episode 0
print(f"\n[Dataset] Loading parquets from {DATASET_ROOT}/data ({len(parquet_files)} files) ...")
import pyarrow as pa
tables = [pq.read_table(f) for f in parquet_files]
full_table = pa.concat_tables(tables)
df = full_table.to_pandas()

print(f"[Dataset] Total rows: {len(df)}, columns: {list(df.columns)}")

ep0 = df[df["episode_index"] == 0].sort_values("frame_index")
state_col = "observation.state"
if state_col not in df.columns:
    candidates = [c for c in df.columns if "state" in c.lower()]
    print(f"[Dataset] '{state_col}' not found. Candidates: {candidates}")
    state_col = candidates[0] if candidates else None

joint_angles = np.array(ep0[state_col].tolist(), dtype=np.float64)
print(f"\n[Dataset] Episode 0: {len(ep0)} frames")
print(f"  joint_angles.shape = {joint_angles.shape}")
print(f"  joint_angles[0]    = {joint_angles[0]}")
print(f"  joint_angles[-1]   = {joint_angles[-1]}")

# ── Step 4: Compute EEF poses ─────────────────────────────────────────────────
T = len(joint_angles)
n_arm_joints = min(5, model.nq)  # 5 arm joints, joint 5 is gripper

eef_positions = np.zeros((T, 3))
eef_rotations = np.zeros((T, 3, 3))

# Dataset records joint angles in degrees; pinocchio expects radians.
DEG2RAD = np.pi / 180.0
print(f"\n[Units] Converting joint angles from degrees to radians (factor={DEG2RAD:.6f})")
print(f"  joint_angles[0] deg:  {joint_angles[0]}")
print(f"  joint_angles[0] rad:  {joint_angles[0] * DEG2RAD}")

for t in range(T):
    q_full = np.zeros(model.nq)
    q_full[:n_arm_joints] = joint_angles[t, :n_arm_joints] * DEG2RAD
    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    pose = data.oMf[frame_id]
    eef_positions[t] = pose.translation.copy()
    eef_rotations[t] = pose.rotation.copy()

# ── Step 5: Sanity checks ─────────────────────────────────────────────────────
print(f"\n=== EEF Position Statistics (frame='{eef_name}') ===")
print(f"Shape: {eef_positions.shape}")
print(f"X range: [{eef_positions[:,0].min():.4f}, {eef_positions[:,0].max():.4f}] m")
print(f"Y range: [{eef_positions[:,1].min():.4f}, {eef_positions[:,1].max():.4f}] m")
print(f"Z range: [{eef_positions[:,2].min():.4f}, {eef_positions[:,2].max():.4f}] m")
print(f"Total XYZ range: {eef_positions.max(0) - eef_positions.min(0)}")
print(f"Mean position:   {eef_positions.mean(0)}")

if eef_positions[:, 2].min() < -0.5:
    print("WARN: EEF goes below -0.5m — possible wrong frame or joint convention")
if eef_positions[:, 2].max() > 1.0:
    print("WARN: EEF above 1.0m — possible wrong frame")
if np.any(np.isnan(eef_positions)):
    print("ERROR: NaN in EEF positions — FK failed")
else:
    print("OK: No NaN values")

deltas = np.diff(eef_positions, axis=0)
print(f"\n=== Step-wise EEF delta statistics ===")
print(f"Mean |step| per axis: {np.abs(deltas).mean(axis=0)} m/step")
print(f"Max  |step| per axis: {np.abs(deltas).max(axis=0)} m/step")
max_step = np.abs(deltas).max()
if max_step > 0.05:
    print(f"WARN: max step {max_step:.4f}m > 5cm — possible joint discontinuity or wrong joint order")
else:
    print(f"OK: max step {max_step:.4f}m < 5cm")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for i, label in enumerate(["X", "Y", "Z"]):
    axes[i].plot(eef_positions[:, i])
    axes[i].set_ylabel(f"EEF {label} (m)")
    axes[i].set_xlabel("timestep")
axes[0].set_title(f"EEF Trajectory — Episode 0 (frame='{eef_name}')")
plt.tight_layout()
out_img = REPO_ROOT / "eef_trajectory_episode0.png"
plt.savefig(out_img)
print(f"\nSaved: {out_img}")

# ── Step 6: Chunk-wise EEF deltas ────────────────────────────────────────────
chunk_size = 32
chunk_deltas_list = []
for t in range(0, len(eef_positions) - chunk_size, chunk_size):
    chunk = eef_positions[t:t + chunk_size]
    ref = eef_positions[t]
    chunk_deltas_list.append(chunk - ref)

if chunk_deltas_list:
    chunk_deltas = np.array(chunk_deltas_list)  # (n_chunks, 32, 3)
    print(f"\n=== Chunk-wise EEF delta statistics (chunk_size={chunk_size}) ===")
    print(f"n_chunks:       {len(chunk_deltas_list)}")
    print(f"Per-axis std:   {chunk_deltas.std(axis=(0, 1))} m")
    print(f"Per-axis max:   {np.abs(chunk_deltas).max(axis=(0, 1))} m")
    print(f"Per-axis p99:   {np.percentile(np.abs(chunk_deltas), 99, axis=(0, 1))} m")

    std_vals = chunk_deltas.std(axis=(0, 1))
    if np.any(std_vals < 0.005):
        print("WARN: chunk delta std < 5mm on some axis — very little EEF motion?")
    elif np.any(std_vals > 0.5):
        print("WARN: chunk delta std > 50cm — FK magnitudes look off")
    else:
        print("OK: chunk delta std in expected range (5mm–50cm)")
else:
    print("WARN: episode too short for any full chunk")
