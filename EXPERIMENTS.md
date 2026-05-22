# Diffusion Policy Experiments — A1 to A5

**Task:** Two-fold towel folding on SO-101.  
**Dataset:** `gaspardthrl/walleed_fold_combined` (~300 episodes, 460K frames).  
**Architecture:** ResNet-18 + spatial softmax + DDPM, horizon=32, n_action_steps=16, n_obs_steps=1.  
**Shared config:** grayworld normalisation, 96×96 input, random crop+affine+colour jitter, DDPM 100 steps (ε-prediction), clip_sample_range=3.0, cosine LR schedule with 500 warmup steps, 100K steps, batch=128.

---

## Core question

Can a diffusion policy for towel folding generalise across starting positions, or does it overfit to absolute joint angles seen during training?  
The A1→A5 ladder isolates the contribution of (a) how proprioception is used, and (b) what action space the policy operates in.

---

## A1 — Full proprio baseline

**Script:** `scripts/train_diffusion_A1.sh`

| Config | Value |
|---|---|
| Action space | Joint relative (6D) |
| State input | All 6 joints |
| Proprio dropout | 0.0 |

**Why:**  
Relative joint actions remove absolute-position bias: the policy predicts *changes* in joint angle rather than absolute targets. Normalised with per-dataset MEAN_STD statistics (`relative_stats.json`). Full proprio gives the policy maximum information about the current configuration.  
This is the upper-bound baseline — best possible state information, simplest action space. If this fails, more ablated variants will too.

**What it tests:** Does relative-action diffusion with full proprio learn a coherent folding policy?

---

## A2 — Gripper-only proprio

**Script:** `scripts/train_diffusion_A2.sh`

| Config | Value |
|---|---|
| Action space | Joint relative (6D) |
| State input | Gripper only (`state_indices=[5]`) |
| Proprio dropout | 0.0 |

**Why:**  
Arm joint angles encode absolute configuration, which can cause the policy to overfit to the exact table position used during data collection. The gripper state (open/close) is useful for contact-phase timing but does not encode absolute arm pose.  
Feeding only the gripper state tests whether removing arm joint feedback forces the policy to rely on vision for position estimation — potentially improving generalisation.

**What it tests:** Does removing arm state (keeping only gripper) improve or hurt performance?

---

## A3 — Proprio dropout

**Script:** `scripts/train_diffusion_A3.sh`

| Config | Value |
|---|---|
| Action space | Joint relative (6D) |
| State input | All 6 joints |
| Proprio dropout | 0.3 |

**Why:**  
A soft alternative to A2. Rather than hard-removing proprio, dropout randomly zeroes the entire proprio vector during training with probability 0.3. This trains a single model that can run with or without proprioception at inference time.  
At deployment the proprio can be set to zero (vision-only inference) or kept active (full inference), giving flexibility without retraining.

**What it tests:** Does dropout regularisation reduce proprio dependence while preserving its benefit when available?

---

## A4 — Vision-only

**Script:** `scripts/train_diffusion_A4.sh`

| Config | Value |
|---|---|
| Action space | Joint relative (6D) |
| State input | None (`state_indices=[]`) |
| Proprio dropout | 0.0 |

**Why:**  
If the policy operates purely from images it cannot overfit to absolute joint configurations — making it the most general joint-space variant. This is the logical extreme of A2/A3.  
In practice, vision-only policies struggle with contact and fine grasping because single-frame images are ambiguous about gripper state. This run measures how much performance is lost by removing all proprio.

**What it tests:** Hard lower bound on proprio — does vision alone suffice for towel folding?

---

## A5 — EEF delta actions

**Script:** `scripts/train_diffusion_A5.sh`

| Config | Value |
|---|---|
| Action space | EEF delta: `[pos_delta(3), rot_6d_delta(6), gripper(1)]` = 10D |
| State input | None |
| Proprio dropout | 0.0 |

**Why:**  
Joint-space relative actions still depend on the arm configuration: the same joint delta produces a different Cartesian displacement depending on where the arm is. EEF delta actions express motion directly in Cartesian/pose space, making the policy's output inherently translation-invariant — the same action means the same gripper displacement regardless of arm pose.  
This is the most principled approach for generalisation to new table positions or object placements.

**Implementation:**
- Pinocchio FK is precomputed for all dataset frames → stored in `eef_poses.npy`
- At training time, `EEFActionProcessorStep` computes chunk-relative EEF deltas on-the-fly and replaces the joint action
- Action: position delta from chunk start (MEAN_STD), 6D rotation delta from chunk start (MEAN_STD), absolute gripper (MIN_MAX)
- At inference: policy outputs 10D EEF delta → unnormalise → IK (placo) → joint angles → robot

**Known limitation:** IK is not implemented for inference deployment yet (the `replay_eef.py` script does this manually but the policy rollout pipeline is missing IK integration).

**What it tests:** Does posing the problem in EEF space improve generalisation to novel starting positions?

---

## B-series — DiT + Flow Matching experiments

**Architecture:** MultiTaskDiT, CLIP-ViT-B/16 vision encoder, flow matching (Euler, 10 steps, β-sampling α=1.5 β=1.0), horizon=32, n_action_steps=24, hidden_dim=512, 6 layers, 8 heads, RoPE.  
**Shared config:** grayworld + grayscale, 224×224 input, cosine LR schedule with 1000 warmup steps, brightness/contrast/sharpness/affine/erasing augmentation.

---

## B1 — DiT FM frozen CLIP

**Script:** `scripts/train_dit_fm_frozen_clip.sh`

| Config | Value |
|---|---|
| Dataset | `gaspardthrl/walleed_teleop_gaspard` (~280 episodes) |
| Action space | Joint relative (default) |
| CLIP LR multiplier | 0.0 (frozen) |
| Steps | 100K |

**Why:**  
With only 280 episodes, fine-tuning a full ViT risks overfitting the vision encoder to dataset-specific appearance (lighting, table texture) or destroying pretrained semantic features with noisy gradients. Freezing CLIP forces the DiT trunk to learn entirely on top of fixed pretrained visual representations.  
This is an ablation of `train_dit_fm.sh` — the only change is `vision_encoder_lr_multiplier=0.0`.

**What it tests:** Does frozen vs. fine-tuned CLIP matter for a small (~280 episode) towel-folding dataset?

---

## B2 — DiT FM absolute + state-history temporal features

**Script:** `scripts/train_dit_fm_absolute_temporal.sh`

| Config | Value |
|---|---|
| Dataset | `gaspardthrl/walleed_fold_combined` (~300 episodes) |
| Action space | Absolute joint positions |
| CLIP LR multiplier | 0.0 (frozen) |
| State input | 6D joints + lag-diff features at t-5, t-10 for motors 0 and 1 → 10D total |
| Steps | 200K |

**Why:**  
Absolute joint actions overfit to the exact arm configuration seen during data collection. Rather than switching to relative or EEF actions, this run augments the state with temporal difference features: `tanh((q[t] - q[t-k]) / scale)` for two "phase-encoding" joints (shoulder_pan=0, shoulder_lift=1) at lags k=5 and k=10 frames.

These lag-diff features encode the *direction and speed of motion* rather than absolute position, giving the policy a sense of what phase the motion is in (reaching, picking, folding, releasing) without requiring it to memorise absolute angles. The shoulders are chosen because they vary most across the four folding phases.

Scales (`state_history_scales.json`) are computed once from the dataset:
```bash
uv run python scripts/compute_state_history_scales.py \
    --repo-id gaspardthrl/walleed_fold_combined --motors 0,1 --lags 5,10
```
The script loads these automatically from `meta/state_history_scales.json` (falls back to HF download if not local).

**What it tests:** Can temporal difference features on the two most phase-discriminative joints compensate for absolute-position overfit, without changing the action space?

---

## Subtask Classifier

**Scripts:** `scripts/annotate_sarm.sh` → `scripts/train_subtask_classifier.sh` / `train_subtask_classifier.py`

### Task

Classify the current frame into one of 8 folding phases:

1. reach the towel first fold
2. pick the towel first fold
3. perform first fold
4. release towel first fold
5. reach the towel second fold
6. pick the towel second fold
7. perform second fold
8. release towel second fold

### Architecture

Frozen backbone + state MLP + previous-subtask embedding → 8-class head. The previous-subtask embedding provides sequential context (phase i is usually followed by phase i+1).

| Input | Dim |
|---|---|
| Image (backbone features) | backbone-dependent |
| Joint state | 6D |
| Previous subtask (embedding) | 8D |

**Backbones evaluated:** `clip`, `dinov2_s`, `dinov2_b`, `resnet18`, `resnet50`

### Workflow

```bash
# 1. Annotate dataset with VLM (GPT-4V labels each frame)
./scripts/annotate_sarm.sh

# 2. Train classifier across all backbones
DATASET_ROOT=~/.cache/.../snapshots/<hash> ./scripts/train_subtask_classifier.sh

# Or single backbone:
BACKBONE=clip DATASET_ROOT=... ./scripts/train_subtask_classifier.sh
```

Training: 30 epochs, batch=128, lr=3e-4, subsampling every 3rd frame, weighted random sampler to handle phase imbalance.

### Why

A subtask classifier enables two downstream uses:
1. **Conditioned policy**: feed the current phase label as an extra input to the main policy, giving it explicit knowledge of which sub-goal to pursue.
2. **Phase-aware evaluation**: automatically detect failure modes per phase rather than treating the full episode as a single success/failure.

---

## Code changes across all runs

### Training infrastructure

| File | Change |
|---|---|
| `configuration_diffusion.py` | Added `image_grayworld`, `image_grayscale` fields |
| `modeling_diffusion.py` | Grayworld white-balance applied in encoder forward pass (before backbone) |
| `processor/eef_action_processor.py` | `EEFActionProcessorStep` + `EEFUnnormalizeProcessorStep`; OOB index clamp at episode boundaries |
| `scripts/precompute_eef_sidecars.py` | Runs pinocchio FK over full dataset, writes `eef_poses.npy` and `eef_stats.json` |

### Hyperparameter fixes

| Parameter | Before | After | Reason |
|---|---|---|---|
| `clip_sample_range` | 2.0 | 3.0 | q99 of normalised joint deltas ≈ 3.0–3.8σ; clipping at 2σ was cutting ~6% of the distribution at inference |
| `image_grayscale` | (was true) | false | Grayworld before grayscale is a near no-op (channel info redistributed then discarded) |

### IK / deployment

| File | Change |
|---|---|
| `model/kinematics.py` | IK iterate solver 10× per step (single call insufficient for orientation convergence) |
| `robots/so_follower/robot_kinematic_processor.py` | Added `orientation_weight` field to `InverseKinematicsEEToJoints` |
| `scripts/replay_eef.py` | Delta-mode EEF replay: torque disable for manual placement, FK-based start pose, position delta with fixed orientation |
| `scripts/move_eef_forward.py` | Utility to move EEF along local/world axis with orientation hold |
