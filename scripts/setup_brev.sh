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
  if [ -n "$GITHUB_PAT" ]; then
    git clone --recurse-submodules --branch rcc/dit-fm \
      "https://${GITHUB_PAT}@github.com/gaspardthrl/diffusing" .
  else
    git clone --recurse-submodules --branch rcc/dit-fm \
      https://github.com/gaspardthrl/diffusing .
  fi
fi

echo "==> Storing git credentials and initialising submodules"
if [ -n "$GITHUB_PAT" ]; then
  git config --global credential.helper store
  echo "https://${GITHUB_PAT}@github.com" > ~/.git-credentials
  # Point lerobot submodule remote to use PAT so fetch works
  git -C third_party/lerobot remote set-url origin \
    "https://${GITHUB_PAT}@github.com/GustaveCharles/lerobot" 2>/dev/null || true
fi
# Force checkout the exact submodule commit rcc/dit-fm points to
git submodule update --init --recursive --force
echo "    lerobot submodule at: $(git -C third_party/lerobot rev-parse --short HEAD)"

echo "==> Installing Python deps"
uv sync --extra inference

echo "==> Installing multi_task_dit extras (transformers + diffusers)"
uv sync --extra multi_task_dit 2>/dev/null || \
  uv pip install transformers diffusers accelerate

echo "==> Installing dataset extras (datasets library)"
uv pip install 'lerobot[dataset]'

echo "==> Verifying CUDA"
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

echo "==> Hugging Face login"
# HF_DATASET_TOKEN: read token for private datasets (e.g. gaspardthrl/walleed_teleop_gaspard)
# HF_TOKEN:         write token for pushing model checkpoints to your own account
# If only HF_TOKEN is provided it is used for both (works when dataset is public).
if [ -n "$HF_DATASET_TOKEN" ]; then
  echo "    Caching dataset read token (HF_DATASET_TOKEN)..."
  uv run hf auth login --token "$HF_DATASET_TOKEN"
  echo "    Write token (HF_TOKEN) will be used for model push."
elif [ -n "$HF_TOKEN" ]; then
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
