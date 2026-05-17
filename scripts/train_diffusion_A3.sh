#!/bin/bash
# A3 — Proprio dropout (full 6D state, p=0.3 dropout during training only)
# Purpose: regularize dependence on proprio, encourage vision-driven fallback.
# At inference the full state is always used (dropout is training-only).
#
# Usage:
#   HF_REPO_ID=user/repo ./scripts/train_diffusion_A3.sh [STEPS]
set -e

STEPS="${1:-100000}"
DATASET_REPO_ID="${DATASET_REPO_ID:-gaspardthrl/walleed_fold_combined}"
OUTPUT_DIR="outputs/diffusion_A3_$(date +%Y%m%d_%H%M%S)"

if [ -n "${HF_REPO_ID}" ]; then
  PUSH_FLAGS="--policy.push_to_hub=true --policy.repo_id=${HF_REPO_ID}"
else
  PUSH_FLAGS="--policy.push_to_hub=false"
fi

if [ -n "${DATASET_ROOT}" ]; then
  DATASET_FLAGS="--dataset.repo_id=${DATASET_REPO_ID} --dataset.root=${DATASET_ROOT}"
else
  DATASET_FLAGS="--dataset.repo_id=${DATASET_REPO_ID}"
fi

STATS_FILE="${DATASET_ROOT:-./data}/meta/relative_stats.json"
if [ ! -f "${STATS_FILE}" ]; then
  echo "Downloading relative_stats.json from HF..."
  uv run python - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download
repo_id = os.environ.get("DATASET_REPO_ID", "gaspardthrl/walleed_fold_combined")
root = os.environ.get("DATASET_ROOT", "./data")
os.makedirs(f"{root}/meta", exist_ok=True)
hf_hub_download(repo_id=repo_id, filename="meta/relative_stats.json", repo_type="dataset", local_dir=root)
PYEOF
fi

if uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DEVICE="cuda"
elif uv run python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
  DEVICE="mps"; export PYTORCH_ENABLE_MPS_FALLBACK=1
else
  DEVICE="cpu"
fi
echo "Using device: $DEVICE"

uv run lerobot-train \
  ${DATASET_FLAGS} \
  --dataset.video_backend=pyav \
  --dataset.stats_override_path="${STATS_FILE}" \
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
  --policy.action_normalization_mode=MEAN_STD \
  \
  `# ── A3: full proprio + dropout p=0.3, relative actions ───────────` \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.action_feature_names='["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]' \
  --policy.state_indices=null \
  --policy.proprio_dropout=0.3 \
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
  --job_name="diffusion-A3-proprio-dropout" \
  --wandb.notes="A3 proprio dropout p=0.3, relative actions, MEAN_STD action norm, ResNet-18, ${STEPS} steps" \
  \
  --output_dir="${OUTPUT_DIR}"

echo "Done. Checkpoint: ${OUTPUT_DIR}"
