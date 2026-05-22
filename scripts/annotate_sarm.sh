#!/bin/bash
# Annotate walleed_fold_combined with 8-stage dense subtask labels using Qwen3-VL.
# Saves temporal_proportions_dense.json to dataset meta and pushes to HF.
#
# Usage:
#   ./scripts/annotate_sarm.sh
#   VLM_MODEL=Qwen/Qwen3-VL-7B-Instruct ./scripts/annotate_sarm.sh   # smaller GPU
#   NUM_VIZ=0 ./scripts/annotate_sarm.sh                              # skip visualizations
set -e

DATASET_REPO_ID="${DATASET_REPO_ID:-gaspardthrl/walleed_fold_combined}"
MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
NUM_VIZ="${NUM_VIZ:-5}"
OUTPUT_DIR="outputs/sarm_viz_$(date +%Y%m%d_%H%M%S)"

DENSE_SUBTASKS="reach the towel first fold, pick the towel first fold, perform first fold, release towel first fold, reach the towel second fold, pick the towel second fold, perform second fold, release towel second fold"

echo "Dataset : ${DATASET_REPO_ID}"
echo "Model   : ${MODEL}"
echo "Stages  : ${DENSE_SUBTASKS}"
echo ""

uv run python third_party/lerobot/src/lerobot/data_processing/sarm_annotations/subtask_annotation.py \
  --repo-id "${DATASET_REPO_ID}" \
  --dense-only \
  --dense-subtasks "${DENSE_SUBTASKS}" \
  --video-key observation.images.front \
  --model "${MODEL}" \
  --device cuda \
  --dtype bfloat16 \
  --skip-existing \
  --push-to-hub \
  --num-visualizations "${NUM_VIZ}" \
  --visualize-type dense \
  --output-dir "${OUTPUT_DIR}"

echo ""
echo "Done. Visualizations: ${OUTPUT_DIR}"
echo "Next: run ./scripts/train_sarm.sh"
