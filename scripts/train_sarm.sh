#!/bin/bash
# Train SARM reward model on walleed dataset (single_stage, no annotations needed)
# ~22 min on Mac M-series for 500 steps. Use 5000 steps on a GPU for production.
set -e

HF_TOKEN=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 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
.venv/bin/lerobot-train \
  --dataset.repo_id=gaspardthrl/walleed \
  --policy.type=sarm \
  --policy.annotation_mode=single_stage \
  --policy.image_key=observation.images.front \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --steps=500 \
  --batch_size=16 \
  --num_workers=2 \
  --log_freq=50 \
  --save_freq=500 \
  --wandb.enable=false \
  --output_dir=outputs/sarm_walleed
