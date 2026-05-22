#!/usr/bin/env python
"""Move the robot EEF forward by X metres in the world frame, ignoring rotation.

Usage:
    .venv/bin/python scripts/move_eef_forward.py --port /dev/tty.usbmodem... --distance 0.05
    .venv/bin/python scripts/move_eef_forward.py --port /dev/tty.usbmodem... --distance 0.05 --axis y
"""

import argparse
import time
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

REPO_ROOT = Path(__file__).parent.parent
URDF_PATH = REPO_ROOT / "so101_new_calib.urdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--distance", type=float, default=0.05, help="Distance to move in metres")
    parser.add_argument("--axis", choices=["x", "y", "z", "local_x", "local_y", "local_z"], default="local_z",
                        help="World axis (x/y/z) or gripper-local axis (local_x/y/z) to move along")
    parser.add_argument("--steps", type=int, default=30, help="Number of IK steps (smoother with more)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--robot-id", default="walleed_follower")
    parser.add_argument("--calibration-dir", default="./calibration")
    args = parser.parse_args()

    robot_config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        use_degrees=True,
        calibration_dir=Path(args.calibration_dir),
    )
    robot = SO101Follower(robot_config)

    kinematics = RobotKinematics(
        urdf_path=str(URDF_PATH),
        target_frame_name="gripper_frame_link",
        joint_names=None,
    )

    robot.connect()
    try:
        obs = robot.get_observation()
        joint_pos = np.array(list(obs.values()), dtype=np.float64)

        # Current EEF pose (4x4)
        T_current = kinematics.forward_kinematics(joint_pos)
        print(f"Current EEF position: {T_current[:3, 3].round(4)}")

        # Compute the direction vector to move along
        R = T_current[:3, :3]
        if args.axis.startswith("local_"):
            col = {"local_x": 0, "local_y": 1, "local_z": 2}[args.axis]
            direction = R[:, col]  # gripper-local axis expressed in world frame
        else:
            axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
            direction = np.zeros(3); direction[axis_idx] = 1.0

        target_pos = T_current[:3, 3] + args.distance * direction
        print(f"Direction (world):  {direction.round(4)}")
        print(f"Target  EEF position: {target_pos.round(4)}")

        # Interpolate in steps for smooth motion
        for i in range(1, args.steps + 1):
            t0 = time.perf_counter()
            alpha = i / args.steps
            T_interp = T_current.copy()
            T_interp[:3, 3] = T_current[:3, 3] + alpha * args.distance * direction

            joint_cmd = kinematics.inverse_kinematics(
                current_joint_pos=joint_pos,
                desired_ee_pose=T_interp,
                position_weight=1.0,
                orientation_weight=1.0,
            )
            robot.send_action(dict(zip(obs.keys(), joint_cmd.tolist())))
            joint_pos = joint_cmd.astype(np.float64)
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

        print("Done.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
