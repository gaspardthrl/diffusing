#!/bin/bash
# Train subtask classifier for each backbone and compare val accuracy.
# Run after annotate_sarm.sh completes.
#
# Usage:
#   DATASET_ROOT=~/.cache/.../snapshots/<hash> ./scripts/train_subtask_classifier.sh
#   BACKBONE=clip DATASET_ROOT=... ./scripts/train_subtask_classifier.sh   # single backbone
set -e

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the annotated dataset snapshot path}"
BACKBONES="${BACKBONE:-clip dinov2_s resnet18}"

for BB in $BACKBONES; do
    echo ""
    echo "=========================================="
    echo " Backbone: ${BB}"
    echo "=========================================="
    OUTPUT_DIR="outputs/subtask_cls_${BB}_$(date +%Y%m%d_%H%M%S)"

    uv run python scripts/train_subtask_classifier.py \
        --dataset-root "${DATASET_ROOT}" \
        --backbone "${BB}" \
        --output-dir "${OUTPUT_DIR}" \
        --epochs 30 \
        --batch-size 128 \
        --lr 3e-4 \
        --num-workers 4 \
        --subsample 3

    echo "Saved: ${OUTPUT_DIR}"
done

echo ""
echo "=== Results summary ==="
for d in outputs/subtask_cls_*/config.json; do
    python3 -c "
import json, sys
c = json.load(open('$d'))
print(f\"{c['backbone']:12s}  val_acc={c['best_val_acc']:.3f}\")
"
done
