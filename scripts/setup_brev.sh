#!/bin/bash
# Bootstrap a Brev GPU instance for training (no robot hardware needed).
# Run this once after provisioning the machine.
#
# Usage: bash scripts/setup_brev.sh
set -e

echo "==> Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

echo "==> Cloning repo with submodules (skip if already cloned)"
if [ ! -f pyproject.toml ]; then
  git clone --recurse-submodules --branch rcc/dit-fm https://github.com/gaspardthrl/diffusing .
fi

echo "==> Initialising submodule (in case it was cloned without --recurse-submodules)"
git submodule update --init --recursive

echo "==> Installing Python deps (inference extras only — no robot hardware)"
uv sync --extra inference

echo "==> Installing multi_task_dit extras (transformers + diffusers)"
uv sync --extra multi_task_dit 2>/dev/null || \
  uv pip install transformers diffusers accelerate

echo "==> Verifying CUDA"
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

echo "==> Hugging Face login (paste your token when prompted, or set HF_TOKEN env var)"
if [ -n "$HF_TOKEN" ]; then
  uv run hf auth login --token "$HF_TOKEN"
else
  uv run hf auth login
fi

echo "==> W&B login (paste your API key when prompted, or set WANDB_API_KEY env var)"
if [ -n "$WANDB_API_KEY" ]; then
  uv run wandb login "$WANDB_API_KEY"
else
  uv run wandb login
fi

echo "==> Setup complete. Run training with:"
echo "    HF_REPO_ID=yourname/walleed-dit-fm-100k bash scripts/train_dit_fm.sh"
