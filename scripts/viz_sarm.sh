#!/bin/bash
# Visualize SARM progress predictions for 10 episodes.
# Output PNGs saved to outputs/sarm_viz/
set -e

HF_TOKEN=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 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
.venv/bin/python -m lerobot.policies.sarm.compute_rabc_weights \
  --dataset-repo-id gaspardthrl/walleed \
  --reward-model-path outputs/sarm_walleed/checkpoints/last/pretrained_model \
  --visualize-only \
  --device mps \
  --num-visualizations 10 \
  --head-mode sparse \
  --output-dir outputs/sarm_viz
