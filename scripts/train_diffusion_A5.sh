#!/bin/bash
# A5 — State-free EEF delta policy (vision-only, EEF action space)
# Purpose: replace joint-space actions with chunk-wise EEF delta actions
#   [eef_pos_delta (3), eef_rot_6d_delta (6), gripper (1)] — 10-D total.
# No proprio at all. Normalisation handled entirely by EEFActionProcessorStep.
#
# PREREQUISITE: run precompute_eef_sidecars.py once before training:
#   DATASET_ROOT=./data_combined uv run python scripts/precompute_eef_sidecars.py
#
# Usage:
#   HF_REPO_ID=user/repo ./scripts/train_diffusion_A5.sh [STEPS]
#   DATASET_ROOT=./data_combined HF_REPO_ID=user/repo ./scripts/train_diffusion_A5.sh
set -e

STEPS="${1:-100000}"
DATASET_REPO_ID="${DATASET_REPO_ID:-gaspardthrl/walleed_fold_combined}"
DATASET_ROOT="${DATASET_ROOT:-./data_combined}"
OUTPUT_DIR="outputs/diffusion_A5_$(date +%Y%m%d_%H%M%S)"

if [ -n "${HF_REPO_ID}" ]; then
  PUSH_FLAGS="--policy.push_to_hub=true --policy.repo_id=${HF_REPO_ID}"
else
  PUSH_FLAGS="--policy.push_to_hub=false"
fi

DATASET_FLAGS="--dataset.repo_id=${DATASET_REPO_ID} --dataset.root=${DATASET_ROOT}"

# ── Locate / download EEF sidecar files ──────────────────────────────────────
EEF_POSES_FILE="${DATASET_ROOT}/meta/eef_poses.npy"
EEF_STATS_FILE="${DATASET_ROOT}/meta/eef_stats.json"

if [ ! -f "${EEF_POSES_FILE}" ] || [ ! -f "${EEF_STATS_FILE}" ]; then
  echo "Downloading EEF sidecar files from HF..."
  uv run python - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download
repo_id = os.environ.get("DATASET_REPO_ID", "gaspardthrl/walleed_fold_combined")
root = os.environ.get("DATASET_ROOT", "./data_combined")
os.makedirs(f"{root}/meta", exist_ok=True)
for filename in ["meta/eef_poses.npy", "meta/eef_stats.json"]:
    hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=root)
    print(f"Downloaded {filename}")
PYEOF
fi

if [ ! -f "${EEF_POSES_FILE}" ] || [ ! -f "${EEF_STATS_FILE}" ]; then
  echo "ERROR: EEF sidecar files still missing after download attempt."
  echo "Run: DATASET_ROOT=${DATASET_ROOT} uv run python scripts/precompute_eef_sidecars.py"
  exit 1
fi

# ── Device detection ──────────────────────────────────────────────────────────
if uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DEVICE="cuda"
elif uv run python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
  DEVICE="mps"; export PYTORCH_ENABLE_MPS_FALLBACK=1
else
  DEVICE="cpu"
fi
echo "Using device: $DEVICE"

# ── Train ─────────────────────────────────────────────────────────────────────
uv run lerobot-train \
  ${DATASET_FLAGS} \
  --dataset.video_backend=pyav \
  \
  --policy.type=diffusion \
  --policy.n_obs_steps=1 \
  --policy.horizon=32 \
  --policy.n_action_steps=16 \
  \
  --policy.vision_backbone=resnet18 \
  --policy.use_group_norm=true \
  --policy.image_grayworld=true \
  --policy.image_grayscale=false \
  --policy.resize_shape="[96,96]" \
  --policy.crop_is_random=true \
  --policy.spatial_softmax_num_keypoints=32 \
  \
  --policy.noise_scheduler_type=DDPM \
  --policy.num_train_timesteps=100 \
  --policy.prediction_type=epsilon \
  --policy.clip_sample=true \
  --policy.clip_sample_range=3.0 \
  \
  `# ── A5: EEF delta actions, no proprio, no relative_stats needed ───` \
  --policy.use_eef_actions=true \
  --policy.eef_poses_path="${EEF_POSES_FILE}" \
  --policy.eef_stats_path="${EEF_STATS_FILE}" \
  --policy.eef_action_dim=10 \
  --policy.state_indices='[]' \
  --policy.proprio_dropout=0.0 \
  \
  ${PUSH_FLAGS} \
  \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_name=cosine \
  --policy.scheduler_warmup_steps=500 \
  --policy.device="${DEVICE}" \
  \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=4 \
  --dataset.image_transforms.random_order=true \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"warmth":{"weight":1.0,"type":"WarmthJitter","kwargs":{"warmth":[-0.15,0.15]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-8,8],"translate":[0.12,0.12]}}}' \
  \
  --steps="${STEPS}" \
  --batch_size=128 \
  --num_workers=8 \
  --log_freq=200 \
  --save_freq=20000 \
  \
  --wandb.enable=true \
  --wandb.project=diffusing \
  --job_name="diffusion-A5-eef-delta" \
  --wandb.notes="A5 EEF delta actions (pos+rot6d+gripper=10D), vision-only, ResNet-18, ${STEPS} steps" \
  \
  --output_dir="${OUTPUT_DIR}"

echo "Done. Checkpoint: ${OUTPUT_DIR}"
echo "Deploy with: uv run python scripts/deploy_eef.py --repo <your-hf-repo>"
