#!/bin/bash
# run_eval_dit_fm.sh
# Evaluates the DiT + Flow Matching policy on the SO-101 for two-fold towel folding.
# Checkpoint: set POLICY_PATH and REVISION below or via environment variables.
#
# Prerequisites: SO-101 arm connected, camera plugged in, .env file present.
#
# Usage:
#   POLICY_PATH=gaspardthrl/walleed-dit-fm REVISION=step-100000 ./run_eval_dit_fm.sh
#
# Controls during rollout:
#   Space — pause / resume    q — quit
set -e

# ── 1. Dependencies ───────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing Python dependencies..."
uv sync --extra robot

# ── 2. HuggingFace login ──────────────────────────────────────────────────────
if ! uv run huggingface-cli whoami &>/dev/null; then
  echo "Not logged in to HuggingFace. Running login..."
  uv run huggingface-cli login
fi

# ── 3. Config ─────────────────────────────────────────────────────────────────
POLICY_PATH="${POLICY_PATH:?Set POLICY_PATH to the HF repo, e.g. gaspardthrl/walleed-dit-fm}"
REVISION="${REVISION:-main}"
TASK="${TASK:-Fold the towel}"
DURATION="${DURATION:-1200}"

source .env  # Load FOLLOWER_PORT, FOLLOWER_ID

# Download checkpoint if not local
if [ ! -d "${POLICY_PATH}" ]; then
  CACHE_TAG="${REVISION:-main}"
  LOCAL_POLICY_DIR="./checkpoints/$(echo "${POLICY_PATH}" | tr '/' '_')@${CACHE_TAG}"
  if [ ! -f "${LOCAL_POLICY_DIR}/config.json" ]; then
    echo "Downloading ${POLICY_PATH}@${CACHE_TAG}..."
    uv run hf download "${POLICY_PATH}" --revision "${REVISION}" --local-dir "${LOCAL_POLICY_DIR}"
  else
    echo "Using cached snapshot at ${LOCAL_POLICY_DIR}"
  fi
  POLICY_PATH="${LOCAL_POLICY_DIR}"
fi

# Device detection
if uv run python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
  DEVICE="mps"; export PYTORCH_ENABLE_MPS_FALLBACK=1; echo "Using MPS"
elif uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DEVICE="cuda"; echo "Using CUDA"
else
  DEVICE="cpu"; echo "Using CPU (slow)"
fi

# ── 4. Run eval ───────────────────────────────────────────────────────────────
echo "Policy: ${POLICY_PATH}  Task: ${TASK}  Duration: ${DURATION}s"

uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path="${POLICY_PATH}" \
  --policy.device="${DEVICE}" \
  --inference.type=sync \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --robot.calibration_dir=./calibration \
  --task="${TASK}" \
  --duration="${DURATION}" \
  --display_data=false
