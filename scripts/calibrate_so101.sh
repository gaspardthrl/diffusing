#!/bin/bash
set -e

source .env  # Load FOLLOWER_PORT, LEADER_PORT, etc.

# Usage: ./calibrate.sh

echo "Calibrating follower arm (port: $FOLLOWER_PORT, id: $FOLLOWER_ID)..."
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
    --robot.calibration_dir=./calibration

echo "Calibrating leader arm (port: $LEADER_PORT, id: $LEADER_ID)..."
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID" \
    --teleop.calibration_dir=./calibration

echo "Calibration complete!"