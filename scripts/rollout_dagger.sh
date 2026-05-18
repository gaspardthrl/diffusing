#!/bin/bash
set -e

source .env 

POLICY_PATH="${1:-gaspardthrl/walleed_dit_fm_absolute}"
REPO_ID="${2:-gaspardthrl/rollout_hil_dataset_gus_2}"
NUM_EPISODES="${3:-50}"
TASK="${4:-Fold the towel}"

lerobot-rollout \
  --strategy.type=dagger \
  --strategy.num_episodes="$NUM_EPISODES" \
  --strategy.record_autonomous=true \
  --strategy.upload_every_n_episodes=5 \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --robot.calibration_dir=./calibration \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --teleop.calibration_dir=./calibration \
  --policy.path="$POLICY_PATH" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.single_task="$TASK" \
  --dataset.fps=30 \
  --dataset.episode_time_s=60 \
  --dataset.num_episodes="$NUM_EPISODES" \
  --interpolation_multiplier=2 \
  --dataset.root=./data \
  --resume=false
