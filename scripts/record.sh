#!/bin/bash
set -e

source .env  # Load FOLLOWER_PORT, LEADER_PORT, etc.

REPO_ID="${1:-gaspardthrl/walleed_teleop_gaspard}"
NUM_EPISODES="${2:-10}"
TASK="${3:-Fold the towel}"
RESUME="${4:-true}"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --robot.calibration_dir=./calibration \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --teleop.calibration_dir=./calibration \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root=./data \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.single_task="$TASK" \
  --dataset.episode_time_s=45 \
  --dataset.reset_time_s=0 \
  --resume="$RESUME" \
  --display_data=true
