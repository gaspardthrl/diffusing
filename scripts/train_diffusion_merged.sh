#!/usr/bin/env bash
#
# Train Diffusion Policy (ResNet18 backbone, absolute actions) on
# gaspardthrl/walleed_fold_combined. Designed to run on a Brev L40S (48 GB VRAM).
#
# Required env vars:
#   HF_TOKEN         write-scoped token for the gaspardthrl HF account
#   WANDB_API_KEY    wandb API key for the gaspardthrl wandb account
#
# Optional overrides (env vars):
#   BATCH_SIZE       default 128
#   STEPS            default 100000
#   SAVE_FREQ        default 10000
#   LOG_FREQ         default 200
#   NUM_WORKERS      default 8
#   SEED             default 42
#   OUTPUT_DIR       default outputs/diffusion-resnet-merged
#
# Brev setup before first run (one-time):
#   cd /workspace/diffusing
#   uv sync --extra robot --extra inference
#   git submodule update --init --recursive
#
# Run:
#   export HF_TOKEN=hf_...
#   export WANDB_API_KEY=...
#   bash scripts/train_diffusion_merged.sh
#
# Notes:
#   - lerobot-train saves checkpoints locally every SAVE_FREQ steps and pushes only
#     the FINAL model to HF. To push intermediates, run scripts/push_checkpoints.py
#     either alongside training or after it finishes.
#   - To resume after a crash, re-run with: --resume=true
#   - Pass any extra lerobot-train flags after the script name: `bash ... -- --foo=bar`.

set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN (write token for gaspardthrl HF account)}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY (wandb API key)}"

# Make tokens visible to all child processes / libs
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_TOKEN="$HF_TOKEN"
export WANDB_API_KEY

# Non-interactive logins (idempotent)
hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true

# ---- knobs ---- #
BATCH_SIZE="${BATCH_SIZE:-128}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/diffusion-resnet-merged}"

DATASET_REPO=gaspardthrl/walleed_fold_combined
MODEL_REPO=gaspardthrl/diffusion-resnet-merged
WANDB_PROJECT=diffusion-resnet-merged
WANDB_ENTITY=gaspardthrl

# Image crop: 95% of 480x640 = 456x608. crop_is_random=true at train, center at eval.
CROP_H=456
CROP_W=608

echo "=================================================================="
echo "Diffusion Policy training"
echo "  dataset:        $DATASET_REPO"
echo "  model repo:     $MODEL_REPO   (final push at end of training)"
echo "  wandb:          $WANDB_ENTITY/$WANDB_PROJECT"
echo "  output dir:     $OUTPUT_DIR"
echo "  backbone:       resnet18 (pretrained on ImageNet)"
echo "  actions:        absolute"
echo "  n_obs_steps:    1"
echo "  horizon:        32   n_action_steps: 24   drop_n_last_frames: 8"
echo "  random crop:    ${CROP_H}x${CROP_W} of 480x640"
echo "  batch / steps:  $BATCH_SIZE / $STEPS    (save every $SAVE_FREQ)"
echo "  device:         cuda + AMP"
echo "=================================================================="

exec lerobot-train \
    --dataset.repo_id="$DATASET_REPO" \
    --policy.type=diffusion \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=DEFAULT \
    --policy.n_obs_steps=1 \
    --policy.horizon=32 \
    --policy.n_action_steps=24 \
    --policy.drop_n_last_frames=8 \
    --policy.crop_shape="[$CROP_H,$CROP_W]" \
    --policy.crop_is_random=true \
    --policy.device=cuda \
    --policy.use_amp=true \
    --policy.push_to_hub=true \
    --policy.repo_id="$MODEL_REPO" \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --save_checkpoint=true \
    --save_freq="$SAVE_FREQ" \
    --log_freq="$LOG_FREQ" \
    --num_workers="$NUM_WORKERS" \
    --seed="$SEED" \
    --wandb.enable=true \
    --wandb.project="$WANDB_PROJECT" \
    --wandb.entity="$WANDB_ENTITY" \
    --output_dir="$OUTPUT_DIR" \
    --job_name=diffusion-resnet-merged \
    "$@"
