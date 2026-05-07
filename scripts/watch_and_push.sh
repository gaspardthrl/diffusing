#!/bin/bash
# Watches the output directory for new checkpoints and pushes them to HF Hub.
# Run this in a separate tmux pane alongside training.
#
# Usage:
#   HF_REPO_ID=Shish999/walleed-dit-fm OUTPUT_DIR=outputs/dit_fm_xxx bash scripts/watch_and_push.sh
#
set -e

if [ -z "$HF_REPO_ID" ]; then
  echo "ERROR: HF_REPO_ID is not set. Example: HF_REPO_ID=Shish999/walleed-dit-fm"
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  # Auto-detect: find the most recent outputs/ subdirectory
  OUTPUT_DIR=$(ls -td outputs/*/ 2>/dev/null | head -1)
  if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: OUTPUT_DIR not set and no outputs/ directory found."
    exit 1
  fi
  echo "Auto-detected output dir: $OUTPUT_DIR"
fi

CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints"
PUSHED_FILE="/tmp/pushed_checkpoints_$(echo $HF_REPO_ID | tr '/' '_').txt"
touch "$PUSHED_FILE"

echo "Watching: $CHECKPOINTS_DIR"
echo "Pushing to: $HF_REPO_ID"
echo "Already pushed: $(cat $PUSHED_FILE | wc -l) checkpoints"
echo ""

while true; do
  if [ -d "$CHECKPOINTS_DIR" ]; then
    for checkpoint in "$CHECKPOINTS_DIR"/*/; do
      name=$(basename "$checkpoint")
      # Skip 'last' symlink and already pushed checkpoints
      if [ "$name" = "last" ]; then continue; fi
      if grep -qx "$name" "$PUSHED_FILE" 2>/dev/null; then continue; fi

      pretrained="$checkpoint/pretrained_model"
      if [ -d "$pretrained" ]; then
        echo "[$(date '+%H:%M:%S')] Pushing $name to $HF_REPO_ID ..."
        uv run hf upload "$HF_REPO_ID" "$pretrained" "$name/pretrained_model" --repo-type model && \
          echo "$name" >> "$PUSHED_FILE" && \
          echo "[$(date '+%H:%M:%S')] Done: $name"
      fi
    done
  fi
  sleep 60
done
