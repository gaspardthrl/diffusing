#!/bin/bash
# DiT + Flow Matching ABSOLUTE — generalization fine-tune.
# Resumes from a previous checkpoint and trains on a pre-merged dataset
# (previous samples + new samples already upsampled and pushed to the Hub).
#
# Usage:
#   RESUME_FROM=outputs/dit_fm_absolute_<ts>/checkpoints/last \
#   ./scripts/train_dit_fm_absolute_generalization.sh 50000
#
# Env vars:
#   RESUME_FROM        checkpoint dir (must contain pretrained_model/)
#   DATASET_REPO_ID    merged dataset on the Hub (default: walleed_fold_combined_with_vincent_random)
#   DATASET_ROOT       local dataset root (optional; falls back to Hub download)
#   HF_REPO_ID         push the resulting policy here when training ends
set -e

STEPS="${1:-50000}"
RESUME_FROM="${RESUME_FROM:?Set RESUME_FROM=outputs/<run>/checkpoints/last (the dir containing pretrained_model/)}"
DATASET_REPO_ID="${DATASET_REPO_ID:-gaspardthrl/walleed_fold_combined_with_vincent_random}"
OUTPUT_DIR="outputs/dit_fm_absolute_generalization_$(date +%Y%m%d_%H%M%S)"

if [ ! -d "${RESUME_FROM}/pretrained_model" ]; then
  echo "ERROR: ${RESUME_FROM}/pretrained_model not found." >&2
  echo "RESUME_FROM should point to a checkpoint dir like outputs/<run>/checkpoints/last" >&2
  exit 1
fi

HF_REPO_ID="${HF_REPO_ID:-gaspardthrl/dit-fm-absolute-generalization-vincent}"
PUSH_FLAGS="--policy.push_to_hub=true --policy.repo_id=${HF_REPO_ID}"

if [ -n "${DATASET_ROOT}" ]; then
  DATASET_FLAGS="--dataset.repo_id=${DATASET_REPO_ID} --dataset.root=${DATASET_ROOT}"
else
  DATASET_FLAGS="--dataset.repo_id=${DATASET_REPO_ID}"
fi

# ── Device ─────────────────────────────────────────────────────────────
if uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DEVICE="cuda"
elif uv run python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
  DEVICE="mps"; export PYTORCH_ENABLE_MPS_FALLBACK=1
else
  DEVICE="cpu"
fi
echo "Using device: $DEVICE"

# ── Fine-tune from checkpoint on merged dataset ────────────────────────
# --policy.path loads the trained weights + architecture from the checkpoint;
# optimizer/scheduler start fresh, which is what we want for generalization fine-tune.
uv run lerobot-train \
  --policy.path="${RESUME_FROM}/pretrained_model" \
  \
  ${DATASET_FLAGS} \
  --dataset.video_backend=pyav \
  \
  ${PUSH_FLAGS} \
  \
  --policy.optimizer_lr=2e-5 \
  --policy.scheduler_name=cosine \
  --policy.scheduler_warmup_steps=500 \
  --policy.device="${DEVICE}" \
  \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=6 \
  --dataset.image_transforms.random_order=true \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"warmth":{"weight":1.0,"type":"WarmthJitter","kwargs":{"warmth":[-0.15,0.15]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-8,8],"translate":[0.12,0.12]}},"erasing":{"weight":0.5,"type":"RandomErasing","kwargs":{"p":0.35,"scale":[0.02,0.1],"ratio":[0.3,3.3],"value":0}}}' \
  \
  --steps="${STEPS}" \
  --batch_size=128 \
  --num_workers=8 \
  --log_freq=200 \
  --save_freq=10000 \
  \
  --wandb.enable=true \
  --wandb.project=diffusing \
  --job_name="dit-fm-absolute-generalization" \
  --wandb.notes="Generalization fine-tune from ${RESUME_FROM} on ${DATASET_REPO_ID}, ${STEPS} steps" \
  \
  --output_dir="${OUTPUT_DIR}"

echo "Done. Checkpoint: ${OUTPUT_DIR}"
