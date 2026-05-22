#!/bin/bash
# run_eval_diffusion.sh
# Evaluates the Diffusion Policy (ResNet-18 + DDPM/DDIM) on the SO-101 for two-fold towel folding.
# Checkpoint: gaspardthrl/diffusion-resnet-merged @ step-100000
#
# Prerequisites:
#   - SO-101 arm connected via USB
#   - Camera plugged in (index 0)
#   - Calibration file at ./calibration/<FOLLOWER_ID>.json
#   - .env file with FOLLOWER_PORT and FOLLOWER_ID (see README)
#
# Usage:
#   ./run_eval_diffusion.sh
#   DURATION=60 ./run_eval_diffusion.sh            # shorter episode
#   POLICY_PATH=gaspardthrl/other-model REVISION=step-50000 ./run_eval_diffusion.sh
#
# Controls during rollout:
#   Space — pause / resume    q — quit
set -e

# ── 1. Install uv if missing ──────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "uv not found — installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ── 2. Install Python dependencies ───────────────────────────────────────────
echo "Syncing dependencies (robot extra)..."
uv sync --extra robot

# ── 3. HuggingFace login ──────────────────────────────────────────────────────
if ! uv run huggingface-cli whoami &>/dev/null; then
  echo "Not logged in to HuggingFace — running login..."
  uv run huggingface-cli login
fi

# ── 4. Run eval (delegates to deploy.sh) ─────────────────────────────────────
export POLICY_PATH="${POLICY_PATH:-gaspardthrl/diffusion-resnet-merged}"
export REVISION="${REVISION:-step-100000}"
export TASK="${TASK:-Fold the towel}"
export DURATION="${DURATION:-1200}"

exec ./scripts/deploy.sh "${POLICY_PATH}"
