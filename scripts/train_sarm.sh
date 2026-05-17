#!/bin/bash
# Train SARM reward model on walleed_fold_combined (dense_only, 8 stages).
# Requires annotate_sarm.sh to have been run first (temporal_proportions_dense.json must exist).
#
# Usage:
#   HF_REPO_ID=user/sarm_walleed_fold ./scripts/train_sarm.sh [STEPS]
#   DATASET_ROOT=./data_combined HF_REPO_ID=user/sarm ./scripts/train_sarm.sh
set -e

STEPS="${1:-100000}"
DATASET_REPO_ID="${DATASET_REPO_ID:-gaspardthrl/walleed_fold_combined}"
OUTPUT_DIR="outputs/sarm_$(date +%Y%m%d_%H%M%S)"

DENSE_SUBTASKS='["reach the towel first fold","pick the towel first fold","perform first fold","release towel first fold","reach the towel second fold","pick the towel second fold","perform second fold","release towel second fold"]'

if [ -n "${HF_REPO_ID}" ]; then
  PUSH_FLAGS="--reward_model.push_to_hub=true --reward_model.repo_id=${HF_REPO_ID}"
else
  PUSH_FLAGS="--reward_model.push_to_hub=false"
fi

if [ -n "${DATASET_ROOT}" ]; then
  DATASET_FLAGS="--dataset.repo_id=${DATASET_REPO_ID} --dataset.root=${DATASET_ROOT}"
  PROPS_FILE="${DATASET_ROOT}/meta/temporal_proportions_dense.json"
else
  DATASET_FLAGS="--dataset.repo_id=${DATASET_REPO_ID}"
  # Proportions JSON lives in the HF cache after annotation + push
  PROPS_FILE="$(uv run python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download('${DATASET_REPO_ID}', repo_type='dataset', allow_patterns=['meta/temporal_proportions_dense.json'])
print(os.path.join(path, 'meta/temporal_proportions_dense.json'))
" 2>/dev/null)"
fi

if [ ! -f "${PROPS_FILE}" ]; then
  echo "ERROR: temporal_proportions_dense.json not found at ${PROPS_FILE}"
  echo "Run ./scripts/annotate_sarm.sh first."
  exit 1
fi

# Extract proportions list in subtask order from the JSON
DENSE_PROPS="$(uv run python - <<PYEOF
import json
subtasks = [
    "reach the towel first fold",
    "pick the towel first fold",
    "perform first fold",
    "release towel first fold",
    "reach the towel second fold",
    "pick the towel second fold",
    "perform second fold",
    "release towel second fold",
]
with open("${PROPS_FILE}") as f:
    props = json.load(f)
values = [props.get(s, 1.0 / len(subtasks)) for s in subtasks]
print(json.dumps(values))
PYEOF
)"

echo "Dataset     : ${DATASET_REPO_ID}"
echo "Proportions : ${DENSE_PROPS}"
echo "Output      : ${OUTPUT_DIR}"
echo ""

if uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DEVICE="cuda"
elif uv run python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
  DEVICE="mps"; export PYTORCH_ENABLE_MPS_FALLBACK=1
else
  DEVICE="cpu"
fi
echo "Device: ${DEVICE}"

uv run lerobot-train \
  ${DATASET_FLAGS} \
  \
  --reward_model.type=sarm \
  --reward_model.annotation_mode=dense_only \
  --reward_model.num_dense_stages=8 \
  --reward_model.dense_subtask_names="${DENSE_SUBTASKS}" \
  --reward_model.dense_temporal_proportions="${DENSE_PROPS}" \
  \
  --reward_model.image_key=observation.images.front \
  --reward_model.image_grayworld=true \
  --reward_model.image_grayscale=true \
  --reward_model.state_key=observation.state \
  \
  --reward_model.n_obs_steps=8 \
  --reward_model.frame_gap=30 \
  --reward_model.max_rewind_steps=4 \
  --reward_model.hidden_dim=768 \
  --reward_model.num_heads=12 \
  --reward_model.num_layers=8 \
  --reward_model.dropout=0.1 \
  --reward_model.rewind_probability=0.8 \
  --reward_model.language_perturbation_probability=0.2 \
  \
  --reward_model.device="${DEVICE}" \
  ${PUSH_FLAGS} \
  \
  --steps="${STEPS}" \
  --batch_size=64 \
  --num_workers=8 \
  --log_freq=200 \
  --save_freq=10000 \
  \
  --wandb.enable=true \
  --wandb.project=diffusing \
  --job_name="sarm-walleed-fold" \
  --wandb.notes="SARM dense_only 8-stage, grayworld+grayscale, ${STEPS} steps" \
  \
  --output_dir="${OUTPUT_DIR}"

echo "Done. Checkpoint: ${OUTPUT_DIR}"
