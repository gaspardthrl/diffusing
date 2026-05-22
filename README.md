# diffusing

Imitation-learning experiments on the SO-101 robot arm for a two-fold towel-folding task.
The codebase wraps a custom fork of [LeRobot](https://github.com/huggingface/lerobot)
(vendored at `third_party/lerobot`, tracking [GustaveCharles/lerobot](https://github.com/GustaveCharles/lerobot))
with shell scripts and a small processor library for dataset recording, policy training, and
hardware deployment.

The central research question is generalisation to new table positions: A1–A5 are an ablation
ladder over action spaces and proprio conditioning; the DiT + Flow Matching track explores a
heavier transformer backbone. See `EXPERIMENTS.md` for the full rationale.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Hardware setup](#hardware-setup)
4. [Recording data](#recording-data)
5. [Training — Diffusion A1–A5](#training--diffusion-a1a5)
6. [Training — DiT + Flow Matching](#training--dit--flow-matching)
7. [Deployment](#deployment)
8. [EEF utilities](#eef-utilities)
9. [Data preprocessing scripts](#data-preprocessing-scripts)
10. [Repo structure](#repo-structure)
11. [Custom lerobot patches](#custom-lerobot-patches)
12. [Known limitations](#known-limitations)

---

## Requirements

- Python 3.12 (enforced in `pyproject.toml`)
- [uv](https://astral.sh/uv) package manager
- SO-101 leader + follower arms (Feetech servos)
- A USB camera (OpenCV-compatible)
- CUDA GPU for training (MPS fallback supported for quick local tests)

---

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<your-username>/diffusing.git
cd diffusing
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 3. Install dependencies

Base dependencies only (no hardware drivers):

```bash
uv sync
```

With robot control (Feetech drivers, dataset tools, multi-task DiT):

```bash
uv sync --extra robot
```

With async inference gRPC stack as well:

```bash
uv sync --extra robot --extra inference
```

`./scripts/setup.sh` combines `uv sync --extra robot` with submodule init and `.env` validation
in one shot — useful on a fresh Brev shell.

### 4. Configure `.env`

Scripts source a `.env` file at the repo root. Create it:

```bash
# .env
FOLLOWER_PORT=/dev/tty.usbmodem5B141136511   # macOS: /dev/tty.usbmodem*, Linux: /dev/ttyACM*
LEADER_PORT=/dev/tty.usbmodem5B141129431
FOLLOWER_ID=walleed_follower
LEADER_ID=walleed_leader
HF_TOKEN=hf_...                              # push datasets / models to Hub
```

To find a serial port: plug the arm in alone and run `ls /dev/tty.usbmodem*` (macOS) or
`ls /dev/ttyACM*` (Linux). The arm IDs are arbitrary labels — they just have to match the
calibration JSONs in `calibration/`.

### 5. IK / placo (optional)

A5 training, `replay_eef.py`, and `move_eef_forward.py` require Pinocchio + placo for FK/IK.
These conflict with the `pin>=3.7` pulled in by the main venv, so install them separately:

```bash
uv pip install "pin==3.4.0" "placo>=0.9.6,<0.9.17"
```

**Important:** after this install, invoke those scripts with `.venv/bin/python`, not `uv run`.
`uv run` resolves `pin` from `pyproject.toml` (>=3.7) and will overwrite the 3.4.0 you just
installed.

```bash
.venv/bin/python scripts/replay_eef.py --port /dev/tty.usbmodem... --episode 0
.venv/bin/python scripts/move_eef_forward.py --port /dev/tty.usbmodem... --distance 0.05
```

---

## Hardware setup

Run these steps the first time you connect the arms, in order.

### 1. Motor setup (once per arm)

Assigns servo IDs and baud rate on each chain. Only needed after a fresh arm or a motor swap.

```bash
./scripts/setup_motors.sh
```

### 2. Calibrate

Records homing offsets and joint limits into `calibration/<id>.json` (e.g.
`calibration/walleed_follower.json`). Re-run only if you re-clock a servo horn or otherwise
change the mechanical zero.

```bash
./scripts/calibrate_so101.sh
```

> If `lerobot-teleoperate` reports a mismatch between motor EEPROM and the JSON file: press
> **ENTER** to push the JSON back to the motors, or **`c`** to recalibrate (which invalidates
> policies trained on the previous frame).

### 3. Teleoperate

Mirrors the leader arm on the follower at ~30 Hz and streams the front camera.

```bash
./scripts/run_teleoperation.sh
```

---

## Recording data

```bash
./scripts/record.sh [REPO_ID] [NUM_EPISODES] ["TASK"] [RESUME] [EPISODE_TIME_S]
# defaults: gaspardthrl/walleed_teleop_gaspard  10  "Fold the towel"  true  60
```

Example:

```bash
./scripts/record.sh YOUR_USERNAME/towel-fold 30 "Fold the towel" true 60
```

Episodes are saved under `./data/` and pushed to the Hugging Face Hub after recording.
The camera is configured as `front: {type: opencv, index_or_path: 0, 640x480, fps: 30}`.

---

## Training — Diffusion A1–A5

**Dataset:** `gaspardthrl/walleed_fold_combined` (~300 episodes, 460K frames)
**Architecture:** ResNet-18 + spatial softmax (32 keypoints) + DDPM, 96×96 input, grayworld
white-balance applied in the encoder, random crop + affine + colour jitter augmentation.

**Shared hyperparameters across all A-variants:**

| Parameter | Value |
|---|---|
| Horizon | 32 |
| n_action_steps | 16 |
| n_obs_steps | 1 |
| Noise scheduler | DDPM, 100 train steps, ε-prediction |
| clip_sample_range | 3.0 |
| LR | 1e-4, cosine schedule, 500 warmup steps |
| Steps | 100K |
| Batch size | 128 |
| Image transforms | brightness, contrast, warmth jitter + random affine (±8°, ±12%) |

### Ablation overview

| Variant | Script | Action space | State input | Proprio dropout |
|---|---|---|---|---|
| A1 | `train_diffusion_A1.sh` | Joint relative (6D) | All 6 joints | 0.0 |
| A2 | `train_diffusion_A2.sh` | Joint relative (6D) | Gripper only (index 5) | 0.0 |
| A3 | `train_diffusion_A3.sh` | Joint relative (6D) | All 6 joints | 0.3 |
| A4 | `train_diffusion_A4.sh` | Joint relative (6D) | None | 0.0 |
| A5 | `train_diffusion_A5.sh` | EEF delta (10D) | None | 0.0 |

**A1** — Upper-bound baseline: relative joint actions + full proprioception. Gate for the rest
of the series.

**A2** — Gripper-only state (index 5). Absolute joint angles encode configuration and can cause
the policy to overfit to the table position seen during collection; removing them forces
vision-driven position estimation.

**A3** — Full state with 30% dropout during training. At inference the full state is always
used. Lets a single checkpoint run in either vision-only or full-state mode without retraining.

**A4** — Vision-only hard lower bound (state_dim=0). No state conditioning pathway in the
model at all.

**A5** — EEF delta action space: `[pos_delta(3), rot_6d_delta(6), gripper(1)]` = 10D.
Joint-space relative actions still depend on arm configuration; EEF deltas are inherently
translation-invariant. **Requires precomputed sidecar files** (see below).

### Running a training

```bash
# A1 with default 100K steps, no Hub push
./scripts/train_diffusion_A1.sh

# A1 with custom steps and Hub push
HF_REPO_ID=yourname/diffusion-A1 ./scripts/train_diffusion_A1.sh 50000

# A1 with local data
DATASET_ROOT=./data_combined HF_REPO_ID=yourname/diffusion-A1 ./scripts/train_diffusion_A1.sh
```

The same pattern applies to A2–A4. A5 additionally requires:

```bash
# 1. Precompute EEF sidecar files (once)
DATASET_ROOT=./data_combined uv run python scripts/precompute_eef_sidecars.py

# 2. Train
DATASET_ROOT=./data_combined HF_REPO_ID=yourname/diffusion-A5 ./scripts/train_diffusion_A5.sh
```

A5 looks for `${DATASET_ROOT}/meta/eef_poses.npy` and `${DATASET_ROOT}/meta/eef_stats.json`.
If they are missing it attempts to download them from the HF dataset repo before failing.

---

## Training — DiT + Flow Matching

Multi-task DiT backbone (CLIP ViT-B/16 vision + text encoder, RoPE, 6 transformer layers,
hidden_dim=512, 8 heads) trained with Flow Matching instead of DDPM.

**Default dataset:** `gaspardthrl/walleed_teleop_gaspard`

```bash
./scripts/train_dit_fm.sh                                    # 100K steps
./scripts/train_dit_fm.sh 50000                              # custom steps
DATASET_ROOT=./data HF_REPO_ID=yourname/dit-fm ./scripts/train_dit_fm.sh
```

Key differences from the diffusion A-series:

| Parameter | DiT FM | Diffusion A-series |
|---|---|---|
| Policy type | `multi_task_dit` | `diffusion` |
| Objective | Flow matching | DDPM (ε-prediction) |
| Inference steps | 10 Euler ODE steps | 5 DDIM steps at deployment |
| Vision encoder | CLIP ViT-B/16 (fine-tuned at 0.1× LR) | ResNet-18 (spatial softmax) |
| Image input | 224×224 (no crop, resize only) | 96×96 + random crop |
| Grayscale | `image_grayscale=true` (train + inference) | `image_grayworld=true` (WB only) |
| Timestep sampling | Beta(1.5, 1.0) | — |
| LR | 2e-5 | 1e-4 |
| Batch size | 64 | 128 |

Image augmentations for DiT FM: brightness, contrast, sharpness jitter + random affine
(±8°, ±12%) + random erasing (p=0.35).

Additional DiT FM variant scripts in `scripts/`:

- `train_dit_fm_absolute.sh` — absolute (not relative) action space
- `train_dit_fm_absolute_generalization.sh` — absolute + generalisation-focused augmentation
- `train_dit_fm_absolute_temporal.sh` — absolute + temporal context (n_obs_steps > 1)
- `train_dit_fm_frozen_clip.sh` — CLIP encoder frozen throughout
- `train_dit_fm_relative.sh` — relative actions variant
- `train_dit_fm_v2.sh` — updated hyperparameters
- `train_dit_ddpm.sh` — DiT backbone with DDPM instead of FM

---

## Deployment

```bash
# Default checkpoint (gaspardthrl/diffusion-resnet-merged @ step-100000)
./scripts/deploy.sh

# Specific HF repo (downloads and caches under ./checkpoints/)
./scripts/deploy.sh yourname/diffusion-A1

# Specific revision (git branch in the HF repo)
REVISION=step-100000 ./scripts/deploy.sh gaspardthrl/diffusion-resnet-merged

# Local checkpoint
./scripts/deploy.sh outputs/diffusion_A1_20250101_120000/checkpoint-100000/pretrained_model
```

The script reads `FOLLOWER_PORT` and `FOLLOWER_ID` from `.env`.

**Controls during rollout:** Space = pause/resume, q = quit.

**Env vars that modify rollout behaviour:**

| Var | Default | Effect |
|---|---|---|
| `TASK` | `"Fold the towel"` | Task string passed to the policy |
| `DURATION` | `1200` | Max episode duration in seconds |
| `JOINT_OFFSET` | `0` | Uniform manual offset (degrees) added on top of per-motor auto-offsets |
| `OFFSET_MOTORS` | `[]` | Motor names whose initial reading is used as a per-motor offset |
| `REVISION` | `step-100000` | HF repo branch to download |

**Inference engine:** uses `--inference.type=sync` with DDIM-5 steps. Do not switch to
`inference.type=rtc` for diffusion — RTC replaces the action queue every inference cycle
(~10 Hz) but `DiffusionPolicy` has no RTC inpainting, causing violent jitter.

**Per-motor joint offsets** (`offset_motors` + `joint_offset`): the `BaseStrategy` reads the
arm's initial joint positions on the first observation and uses each selected motor's reading
as an individual offset. This lets you place the arm at a rotated table position and have the
policy see training-like joint values. `joint_offset` is an optional uniform correction on top.

---

## EEF utilities

Both scripts require placo IK (see [IK / placo](#5-ik--placo-optional) above) and must be
invoked with `.venv/bin/python`, not `uv run`.

### `scripts/replay_eef.py`

Replays a recorded episode using precomputed EEF poses + IK instead of raw joint angles.
Useful for verifying delta-mode replay and testing IK quality.

```bash
.venv/bin/python scripts/replay_eef.py \
    --port /dev/tty.usbmodem... \
    --episode 0 \
    --dataset-root ./data_combined \
    --fps 30
```

Requires `data_combined/meta/eef_poses.npy` (generated by `precompute_eef_sidecars.py`).
The URDF loaded is `so101_new_calib.urdf` at the repo root.

### `scripts/move_eef_forward.py`

Moves the EEF a fixed distance along a world or gripper-local axis, holding orientation.
Useful for testing IK and placing the arm before a policy rollout.

```bash
.venv/bin/python scripts/move_eef_forward.py \
    --port /dev/tty.usbmodem... \
    --distance 0.05 \
    --axis local_z   # world: x/y/z  gripper-local: local_x/local_y/local_z
    --steps 30
```

---

## Data preprocessing scripts

All scripts live in `scripts/` and are invoked with `uv run python scripts/<name>.py` unless
otherwise noted.

| Script | Purpose |
|---|---|
| `precompute_eef_sidecars.py` | Pinocchio FK over full dataset → `eef_poses.npy` (shape N×12) + `eef_stats.json`. Required before A5 training. |
| `recompute_stats_relative.py` | Compute relative-action normalisation stats → `relative_stats.json`. Required before A1–A4 training on a new dataset. |
| `compute_eef_fk.py` | Single-episode FK sanity check + visualisation. |
| `compute_state_history_scales.py` | Compute per-joint scale factors for state history normalisation → `state_history_scales.json`. |
| `fix_episode_indices.py` | Repair episode index columns in parquet files (e.g. after partial re-recording). |
| `fix_video_file_indices.py` | Rename video files to match corrected episode indices. |
| `fix_video_timestamps.py` | Rewrite video timestamps after episode repairs. |
| `truncate_at_second_grip.py` | Trim episodes at the second gripper-close event (removes post-fold noise). |
| `count_gripper_segments.py` | Count gripper open/close segments per episode; used to audit dataset quality. |
| `precompute_prev_subtask.py` | Pre-label subtask phases for subtask-conditioned models. |
| `visualize_preprocessing.py` | Visualise raw vs. preprocessed actions for a sample batch. |
| `diagnose_relative_actions.py` | Check that relative-action stats are consistent with the dataset. |

Usage pattern:

```bash
# Relative stats
DATASET_ROOT=./data_combined uv run python scripts/recompute_stats_relative.py

# EEF sidecars (A5 prerequisite)
DATASET_ROOT=./data_combined uv run python scripts/precompute_eef_sidecars.py
# Skip Hub upload:
DATASET_ROOT=./data_combined uv run python scripts/precompute_eef_sidecars.py --no-upload
```

---

## Repo structure

```
diffusing/
├── calibration/                   # Per-arm homing-offset JSONs (committed)
│   ├── walleed_follower.json
│   └── walleed_leader.json
├── scripts/
│   ├── setup.sh                   # uv sync + .env validation
│   ├── setup_motors.sh            # One-time servo ID/baud setup
│   ├── calibrate_so101.sh         # Per-arm calibration
│   ├── run_teleoperation.sh       # Leader→follower mirror + camera
│   ├── record.sh                  # Record teleop episodes and push to Hub
│   ├── deploy.sh                  # Deploy diffusion policy on hardware
│   ├── train_diffusion_A{1..5}.sh # Ablation training scripts
│   ├── train_dit_fm.sh            # DiT + Flow Matching training
│   ├── train_dit_fm_*.sh          # DiT FM variant scripts
│   ├── replay_eef.py              # EEF-IK episode replay
│   ├── move_eef_forward.py        # Move EEF along an axis via IK
│   ├── precompute_eef_sidecars.py # FK precomputation (A5 prerequisite)
│   ├── recompute_stats_relative.py# Relative-action stats computation
│   └── [dataset repair / curation scripts — see table above]
├── src/
│   └── lerobot_policy_diffusing/  # Project-specific Python (currently minimal)
├── third_party/
│   └── lerobot/                   # Custom lerobot fork (git submodule)
├── so101_new_calib.urdf           # Calibrated SO-101 URDF (required for IK)
├── state_history_scales.json      # Per-joint scale factors for state history
├── pyproject.toml
└── outputs/                       # Training outputs (gitignored)
```

---

## Custom lerobot patches

The `third_party/lerobot` submodule tracks `GustaveCharles/lerobot` and contains the following
changes on top of upstream HuggingFace LeRobot:

### Diffusion policy

| File | Change |
|---|---|
| `policies/diffusion/configuration_diffusion.py` | Added `image_grayworld` and `image_grayscale` config fields |
| `policies/diffusion/modeling_diffusion.py` | Grayworld white-balance applied in the encoder forward pass (before the ResNet backbone) when `image_grayworld=True`; grayscale conversion when `image_grayscale=True` |

### Processor pipeline

| File | Change |
|---|---|
| `processor/eef_action_processor.py` | `EEFActionProcessorStep` — loads `eef_poses.npy` sidecar, computes chunk-relative EEF deltas (pos 3D + rot 6D + gripper 1D = 10D) on-the-fly, applies MEAN_STD / MIN_MAX normalisation, replaces `batch["action"]`. `EEFUnnormalizeProcessorStep` inverts normalisation at inference. OOB global index clamp at episode boundaries. |
| `processor/relative_action_processor.py` | `RelativeActionsProcessorStep` (joint relative: action − state) and `AbsoluteActionsProcessorStep` (inverse). |

### Rollout

| File | Change |
|---|---|
| `rollout/context.py` | Removed the `SyncInferenceEngine` guard that blocked relative-action processor steps. The processor pipeline now handles the relative↔absolute conversion correctly for diffusion. |
| `rollout/strategies/base.py` | Per-motor joint offsets: on the first observation, each motor in `offset_motors` has its initial reading recorded as an individual offset. This is applied symmetrically to observations (subtract) and actions (add), enabling deployment at a rotated table position without retraining. |

### Kinematics / IK

| File | Change |
|---|---|
| `model/kinematics.py` | IK solver runs 10 iterations per step (single call was insufficient for orientation convergence) |
| `robots/so_follower/robot_kinematic_processor.py` | `orientation_weight` field added to `InverseKinematicsEEToJoints` config |

---

## Known limitations

- **A5 inference not fully wired.** `EEFUnnormalizeProcessorStep` outputs unnormalised EEF
  deltas at inference time; the IK step that converts those back to joint angles is not
  integrated into the rollout pipeline. `replay_eef.py` does this conversion manually.
  To deploy an A5 policy on hardware, the IK call must be added to the processor pipeline.

- **placo incompatible with `uv run`.** Installing `pin==3.4.0` + placo into the venv and
  then using `uv run` will silently replace pin 3.4.0 with the >=3.7 version from
  `pyproject.toml`. Always use `.venv/bin/python` for scripts that need placo.

- **Calibration mismatch.** If you see `"Mismatch between calibration values in the motor
  and the calibration file"` at startup, the servo EEPROM diverged from the JSON. Press
  ENTER to push the JSON back, or press `c` to recalibrate (invalidates existing policies).

- **Camera latency on macOS.** If the camera stream falls behind (`OpenCVCamera(0) latest
  frame is too old`), try adding `fourcc: MJPG` to the camera config, using an external USB
  webcam, or a different USB port. macOS Continuity Camera can grab the device index silently.
