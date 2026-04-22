#!/bin/bash
set -e

source .env  # Load FOLLOWER_PORT, LEADER_PORT, etc.

lerobot-setup-motors \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT"

lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT"