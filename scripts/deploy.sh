#!/bin/bash
# Deploy a trained diffusion policy on the SO-101 arm (runs locally on Mac/Linux).
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

POLICY_PATH="${1:-gaspardthrl/diffusion-resnet-merged}"
REVISION="${REVISION:-step-100000}"
TASK="${TASK:-Fold the towel}"
DURATION="${DURATION:-1200}"   # seconds per episode
JOINT_OFFSET="${JOINT_OFFSET:-0}"        # offset in degrees applied to OFFSET_MOTORS
OFFSET_MOTORS="${OFFSET_MOTORS:-[]}"  # motors 0 and 4 by default
# NOTE: joint_offset / offset_motors are NOT on this lerobot branch
# (gc/subtask-classifier-cond). They exist on feat/diffusion-grayworld-grayscale.
# Re-enable below when switching branches.

# --policy.revision isn't exposed as a draccus CLI flag (only as a
# from_pretrained() kwarg), so passing it does nothing. Pre-download the
# requested HF branch to a local dir and point --policy.path at it.
# Override branch with: REVISION=step-N ./scripts/deploy.sh
if [ ! -d "${POLICY_PATH}" ]; then
  CACHE_TAG="${REVISION:-main}"
  LOCAL_POLICY_DIR="./checkpoints/$(echo "${POLICY_PATH}" | tr '/' '_')@${CACHE_TAG}"
  if [ ! -f "${LOCAL_POLICY_DIR}/config.json" ] || find "${LOCAL_POLICY_DIR}" -name "*.incomplete" -type f -print -quit 2>/dev/null | grep -q .; then
    [ -d "${LOCAL_POLICY_DIR}" ] && echo "Partial cache at ${LOCAL_POLICY_DIR}, removing" && rm -rf "${LOCAL_POLICY_DIR}"
    echo "Downloading ${POLICY_PATH}@${CACHE_TAG} -> ${LOCAL_POLICY_DIR}"
    if [ -n "${REVISION}" ]; then
      uv run hf download "${POLICY_PATH}" --revision "${REVISION}" --local-dir "${LOCAL_POLICY_DIR}"
    else
      uv run hf download "${POLICY_PATH}" --local-dir "${LOCAL_POLICY_DIR}"
    fi
  else
    echo "Using cached snapshot at ${LOCAL_POLICY_DIR}"
  fi
  POLICY_PATH="${LOCAL_POLICY_DIR}"
fi

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
  --strategy.offset_motors="${OFFSET_MOTORS}" \
  \
  `# ── Policy ──────────────────────────────────────────────────────────` \
  --policy.path="${POLICY_PATH}" \
  --policy.device="${DEVICE}" \
  \
  `# DDPM-100 is slow on MPS; DDIM-10 is a compromise (5 steps was very noisy).` \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=5 \
  \
  `# ── Inference backend ───────────────────────────────────────────────` \
  `# Sync + chunk FIFO: drain one full chunk (correct relative anchor), then replan.` \
  `# Do NOT use inference.type=rtc for diffusion — rtc.enabled replaces the queue` \
  `# every inference (~10 Hz) but DiffusionPolicy has no RTC inpainting → violent jitter.` \
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
  --display_data=false
