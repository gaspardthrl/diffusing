#!/bin/bash
# DiT + Flow Matching with ABSOLUTE actions + STATE-HISTORY temporal features.
#   - CLIP vision encoder frozen
#   - Grayworld + warmth augmentation
#   - Absolute joint positions
#   - State input augmented with tanh-scaled lag-diff features for 2 phase-encoding motors at t-5 and t-10
#
# State-history scales loaded automatically from meta/state_history_scales.json (local or HF).
# Generate once with:
#   uv run python scripts/compute_state_history_scales.py \
#       --repo-id gaspardthrl/walleed_fold_combined --motors 0,1 --lags 5,10
#
# Usage:
#   ./scripts/train_dit_fm_absolute_temporal.sh                 # default 200k steps
#   ./scripts/train_dit_fm_absolute_temporal.sh 300000
#   DATASET_ROOT=./data ./scripts/train_dit_fm_absolute_temporal.sh
#   HF_REPO_ID=user/repo ./scripts/train_dit_fm_absolute_temporal.sh
set -e

STEPS="${1:-200000}"
DATASET_REPO_ID="${DATASET_REPO_ID:-gaspardthrl/walleed_fold_combined}"
OUTPUT_DIR="outputs/dit_fm_absolute_temporal_$(date +%Y%m%d_%H%M%S)"

# ──────────────────────────────────────────────────────────────────────
# State-history config — loaded automatically from state_history_scales.json
# Generate with:
#   uv run python scripts/compute_state_history_scales.py \
#       --repo-id gaspardthrl/walleed_fold_combined --motors 0,1 --lags 5,10
# ──────────────────────────────────────────────────────────────────────
SCALES_FILE="${DATASET_ROOT:-./data}/meta/state_history_scales.json"
if [ ! -f "${SCALES_FILE}" ]; then
  echo "state_history_scales.json not found locally, downloading from HF..."
  uv run python - <<'PYEOF'
import os, json
from huggingface_hub import hf_hub_download
repo_id = os.environ.get("DATASET_REPO_ID", "gaspardthrl/walleed_fold_combined")
root = os.environ.get("DATASET_ROOT", "./data")
dest = f"{root}/meta/state_history_scales.json"
os.makedirs(f"{root}/meta", exist_ok=True)
path = hf_hub_download(repo_id=repo_id, filename="meta/state_history_scales.json", repo_type="dataset", local_dir=root)
print(f"Downloaded to {path}")
PYEOF
fi

STATE_HISTORY_MOTORS=$(uv run python -c "
import json; d=json.load(open('${SCALES_FILE}'))
print(str(d['motors']).replace(' ',''))
")
STATE_HISTORY_LAGS=$(uv run python -c "
import json; d=json.load(open('${SCALES_FILE}'))
print(str(d['lags']).replace(' ',''))
")
STATE_HISTORY_SCALES=$(uv run python -c "
import json; d=json.load(open('${SCALES_FILE}'))
print('['+','.join(f\"{s:.4f}\" for s in d['scales'])+']')
")
echo "State history — motors: ${STATE_HISTORY_MOTORS}  lags: ${STATE_HISTORY_LAGS}  scales: ${STATE_HISTORY_SCALES}"
# ──────────────────────────────────────────────────────────────────────

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
  \
  --policy.type=multi_task_dit \
  --policy.objective=flow_matching \
  --policy.num_integration_steps=10 \
  --policy.integration_method=euler \
  --policy.timestep_sampling_strategy=beta \
  --policy.timestep_sampling_alpha=1.5 \
  --policy.timestep_sampling_beta=1.0 \
  --policy.sigma_min=0.0 \
  \
  --policy.n_obs_steps=1 \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  \
  --policy.hidden_dim=512 \
  --policy.num_layers=6 \
  --policy.num_heads=8 \
  --policy.use_rope=true \
  \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.text_encoder_name=openai/clip-vit-base-patch16 \
  --policy.vision_encoder_lr_multiplier=0.0 \
  --policy.image_resize_shape="[224,224]" \
  --policy.image_grayscale=true \
  --policy.image_grayworld=true \
  \
  `# ── ABSOLUTE actions ──────────────────────────────────────────────` \
  --policy.use_relative_actions=false \
  --policy.action_feature_names='["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]' \
  --policy.default_task="Fold the towel" \
  \
  `# ── STATE-HISTORY temporal features ───────────────────────────────` \
  `# State input grows from 6D to 6 + len(motors)*len(lags) = 10D.` \
  --policy.state_history_motors="${STATE_HISTORY_MOTORS}" \
  --policy.state_history_lags="${STATE_HISTORY_LAGS}" \
  --policy.state_history_scales="${STATE_HISTORY_SCALES}" \
  \
  ${PUSH_FLAGS} \
  \
  --policy.optimizer_lr=2e-5 \
  --policy.scheduler_name=cosine \
  --policy.scheduler_warmup_steps=1000 \
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
  --save_freq=20000 \
  --policy.device="${DEVICE}" \
  \
  --wandb.enable=true \
  --wandb.project=diffusing \
  --job_name="dit-fm-absolute-temporal" \
  --wandb.notes="DiT FM ${STEPS} steps, ABSOLUTE actions + state history (lags ${STATE_HISTORY_LAGS}), frozen CLIP" \
  \
  --output_dir="${OUTPUT_DIR}"

echo "Done. Checkpoint: ${OUTPUT_DIR}"
