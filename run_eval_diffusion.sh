#!/bin/bash
# run_eval_diffusion.sh
# Evaluates the diffusion policy (ResNet-18 + DDPM) on the SO-101 for two-fold towel folding.
# Checkpoint: gaspardthrl/diffusion-resnet-merged @ step-100000
#
# Prerequisites: SO-101 arm connected, camera plugged in, .env file present.
#
# Usage:
#   ./run_eval_diffusion.sh                  # uses defaults below
#   DURATION=60 ./run_eval_diffusion.sh      # shorter episode
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

# ── 2. HuggingFace login (needed to download private model weights) ────────────
if ! uv run huggingface-cli whoami &>/dev/null; then
  echo "Not logged in to HuggingFace. Running login..."
  uv run huggingface-cli login
fi

# ── 3. Run eval ───────────────────────────────────────────────────────────────
POLICY_PATH="${POLICY_PATH:-gaspardthrl/diffusion-resnet-merged}"
REVISION="${REVISION:-step-100000}"
TASK="${TASK:-Fold the towel}"
DURATION="${DURATION:-1200}"

REVISION="${REVISION}" POLICY_PATH="${POLICY_PATH}" TASK="${TASK}" DURATION="${DURATION}" \
  ./scripts/deploy.sh "${POLICY_PATH}"
