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
