#!/bin/bash
# Train DiT policy with Flow Matching on the walleed dataset.
#
# Designed to run on Brev (GPU). Estimated ~2-3h on a single L40S for 100k steps.
#
# Usage:
#   ./scripts/train_dit_fm.sh                        # default 100k steps
#   ./scripts/train_dit_fm.sh 50000                  # custom step count
#   DATASET_ROOT=./data ./scripts/train_dit_fm.sh    # use local data folder
#   HF_REPO_ID=yourname/walleed-dit ./scripts/train_dit_fm.sh  # push to Hub
#
set -e

STEPS="${1:-100000}"
DATASET_REPO_ID="${DATASET_REPO_ID:-gaspardthrl/walleed_teleop_gaspard}"
OUTPUT_DIR="outputs/dit_fm_$(date +%Y%m%d_%H%M%S)"

# Optional: push checkpoint to HF Hub after training
if [ -n "${HF_REPO_ID}" ]; then
  PUSH_FLAGS="--policy.push_to_hub=true --policy.repo_id=${HF_REPO_ID}"
  echo "Will push model to HF Hub: ${HF_REPO_ID}"
else
  PUSH_FLAGS="--policy.push_to_hub=false"
fi

# Optional: use local data folder instead of streaming from Hub
if [ -n "${DATASET_ROOT}" ]; then
  DATASET_FLAGS="--dataset.repo_id=${DATASET_REPO_ID} --dataset.root=${DATASET_ROOT}"
  echo "Using local dataset at: ${DATASET_ROOT}"
else
  DATASET_FLAGS="--dataset.repo_id=${DATASET_REPO_ID}"
fi

# On Brev use cuda; fall back to mps on Apple Silicon for quick local tests
if uv run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DEVICE="cuda"
elif uv run python -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
  DEVICE="mps"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
else
  DEVICE="cpu"
fi
echo "Using device: $DEVICE"

uv run lerobot-train \
  \
  `# ── Dataset ───────────────────────────────────────────────────────` \
  ${DATASET_FLAGS} \
  --dataset.video_backend=pyav \
  \
  `# ── Policy: DiT + Flow Matching ───────────────────────────────────` \
  --policy.type=multi_task_dit \
  --policy.objective=flow_matching \
  \
  `# FM inference: 10 steps is enough for straight-path ODE (vs 100 default)` \
  --policy.num_integration_steps=10 \
  --policy.integration_method=euler \
  --policy.timestep_sampling_strategy=beta \
  --policy.timestep_sampling_alpha=1.5 \
  --policy.timestep_sampling_beta=1.0 \
  --policy.sigma_min=0.0 \
  \
  `# ── Temporal context ──────────────────────────────────────────────` \
  --policy.n_obs_steps=1 \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  \
  `# ── Transformer architecture ──────────────────────────────────────` \
  --policy.hidden_dim=512 \
  --policy.num_layers=6 \
  --policy.num_heads=8 \
  --policy.use_rope=true \
  \
  `# ── Vision/Text backbone (CLIP ViT-B/16) ──────────────────────────` \
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.text_encoder_name=openai/clip-vit-base-patch16 \
  --policy.vision_encoder_lr_multiplier=0.1 \
  `# Resize full scene directly to CLIP's required 224×224 — no crop, no lost corners` \
  --policy.image_resize_shape="[224,224]" \
  `# Grayscale applied inside the model — runs at BOTH train and inference time` \
  --policy.image_grayscale=true \
  \
  ${PUSH_FLAGS} \
  \
  `# ── Optimizer ─────────────────────────────────────────────────────` \
  --policy.optimizer_lr=2e-5 \
  --policy.scheduler_name=cosine \
  --policy.scheduler_warmup_steps=1000 \
  \
  `# ── Image augmentations ────────────────────────────────────────────` \
  `# Grayscale is handled by --policy.image_grayscale (runs at train+inference).` \
  `# These 5 augmentations add robustness on top of the grayscale image.` \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=5 \
  --dataset.image_transforms.random_order=true \
  --dataset.image_transforms.tfs='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-8,8],"translate":[0.12,0.12]}},"erasing":{"weight":0.5,"type":"RandomErasing","kwargs":{"p":0.35,"scale":[0.02,0.1],"ratio":[0.3,3.3],"value":0}}}' \
  \
  `# ── Training loop ─────────────────────────────────────────────────` \
  --steps="${STEPS}" \
  --batch_size=64 \
  --num_workers=8 \
  --log_freq=200 \
  --save_freq=10000 \
  --policy.device="${DEVICE}" \
  \
  `# ── Logging ───────────────────────────────────────────────────────` \
  --wandb.enable=true \
  --wandb.project=diffusing \
  --wandb.run_name="dit-fm-280ep" \
  --wandb.notes="DiT FM ${STEPS} steps, 280ep, grayscale+aug, n_obs=1, 10 ODE steps" \
  \
  --output_dir="${OUTPUT_DIR}"

echo "Training complete. Checkpoint saved to: ${OUTPUT_DIR}"
