#!/bin/bash
# Deploy a trained DiT + FM policy on the SO-101 arm (runs locally on Mac/Linux).
#
# Usage:
#   ./scripts/deploy.sh                                          # use default checkpoint
#   ./scripts/deploy.sh Shish999/walleed-dit-fm-100k             # HF repo (downloads if needed)
#   ./scripts/deploy.sh outputs/walleed-dit-fm-50k/checkpoint-50000/pretrained_model  # local path
#
# Controls during rollout:
#   Space   — pause / resume
#   q       — quit
#
set -e

source .env  # Load FOLLOWER_PORT, FOLLOWER_ID etc.

POLICY_PATH="${1:-Shish999/walleed-dit-fm-100k}"
TASK="${TASK:-Fold the towel}"
DURATION="${DURATION:-60}"   # seconds per episode

# Use MPS on Apple Silicon, CUDA on Linux GPU, else CPU
if uv run python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
  DEVICE="mps"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  echo "Using MPS (Apple Silicon)"
elif uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DEVICE="cuda"
  echo "Using CUDA"
else
  DEVICE="cpu"
  echo "Using CPU (slow)"
fi

echo "Policy : ${POLICY_PATH}"
echo "Task   : ${TASK}"
echo "Duration: ${DURATION}s per episode"
echo ""

uv run lerobot-rollout \
  \
  `# ── Strategy ────────────────────────────────────────────────────────` \
  --strategy.type=base \
  \
  `# ── Policy ──────────────────────────────────────────────────────────` \
  --policy.path="${POLICY_PATH}" \
  --policy.device="${DEVICE}" \
  \
  `# ── Inference backend ───────────────────────────────────────────────` \
  `# sync: one policy call per control tick — safe for DiT on MPS (~200ms)` \
  --inference.type=sync \
  \
  `# ── Robot ───────────────────────────────────────────────────────────` \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --robot.calibration_dir=./calibration \
  \
  `# ── Task + duration ─────────────────────────────────────────────────` \
  --task="${TASK}" \
  --duration="${DURATION}" \
  \
  `# ── Display live feed in a window ───────────────────────────────────` \
  --display_data=true
