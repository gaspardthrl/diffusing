#!/usr/bin/env python
"""Replay a recorded episode on the SO-101 using precomputed EEF poses + IK.

Instead of sending recorded joint angles directly, this script looks up the
precomputed absolute EEF pose for each frame (from eef_poses.npy), solves IK,
and sends the resulting joint angles to the robot.

Prerequisites:
    Run precompute_eef_sidecars.py once to generate eef_poses.npy:
        DATASET_ROOT=./data_combined uv run python scripts/precompute_eef_sidecars.py

Usage:
    uv run python scripts/replay_eef.py --port /dev/tty.usbmodem... --episode 0
    uv run python scripts/replay_eef.py --port /dev/tty.usbmodem... --episode 3 \\
        --dataset-root ./data_combined --fps 30
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import InverseKinematicsEEToJoints
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation

REPO_ROOT = Path(__file__).parent.parent
URDF_PATH = REPO_ROOT / "so101_new_calib.urdf"
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_episode(dataset_root: Path, episode_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Load global frame indices and gripper values for one episode.

    Returns:
        global_indices: (T,) int array of global dataset row indices, in time order
        gripper_values: (T,) float array of absolute gripper angles (degrees)
    """
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset_root}/data")

    table = pa.concat_tables([pq.read_table(f) for f in parquet_files])
    df = table.to_pandas()

    ep_df = df[df["episode_index"] == episode_idx].sort_values("index")
    if ep_df.empty:
        raise ValueError(f"Episode {episode_idx} not found (available: {sorted(df['episode_index'].unique())})")

    global_indices = ep_df["index"].to_numpy()
    states = np.array(ep_df["observation.state"].tolist(), dtype=np.float32)
    gripper_values = states[:, 5]  # joint index 5 = gripper (absolute degrees)

    return global_indices, gripper_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodem5A460814411")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--dataset-root", default="./data_combined")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--robot-id", default="walleed_follower", help="Robot id matching the calibration filename (without .json)")
    parser.add_argument("--calibration-dir", default="./calibration", help="Directory containing <robot-id>.json calibration file")
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=0.1,
        help="IK orientation weight (0=position-only, higher=stricter orientation tracking)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    eef_poses_path = dataset_root / "meta" / "eef_poses.npy"

    if not eef_poses_path.exists():
        raise FileNotFoundError(
            f"EEF poses not found at {eef_poses_path}.\n"
            "Run: DATASET_ROOT=./data_combined uv run python scripts/precompute_eef_sidecars.py"
        )

    print(f"Loading EEF poses from {eef_poses_path} ...")
    eef_poses = np.load(eef_poses_path)  # (N_total, 12): [pos(3), rotmat_flat(9)]

    print(f"Loading episode {args.episode} from {dataset_root} ...")
    global_indices, gripper_values = load_episode(dataset_root, args.episode)
    n_frames = len(global_indices)
    print(f"Episode {args.episode}: {n_frames} frames @ {args.fps} fps")

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
        joint_names=list(robot.bus.motors.keys()),
    )

    ik_pipeline = RobotProcessorPipeline(
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=list(robot.bus.motors.keys()),
                # Open-loop replay: chain IK solutions rather than re-reading
                # current joints each step (avoids sensor noise accumulation).
                initial_guess_current_joints=False,
                orientation_weight=args.orientation_weight,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot.connect()
    try:
        # Episode reference pose (first frame)
        ep_ref_pose = eef_poses[int(global_indices[0])]
        ep_ref_pos = ep_ref_pose[:3]
        ep_ref_R   = ep_ref_pose[3:].reshape(3, 3)

        print(f"\nEpisode ref pos: {ep_ref_pos.round(4)}")
        print("Releasing motors — place the arm at your desired starting position, then press ENTER.")
        robot.bus.disable_torque()
        input()

        # Read position BEFORE re-enabling torque so servos haven't snapped back yet
        robot_obs0 = robot.get_observation()
        motor_keys = {k.replace(".pos", ""): v for k, v in robot_obs0.items() if k.endswith(".pos")}
        q0 = np.array([motor_keys[n] for n in kinematics.joint_names], dtype=np.float64)
        T_start = kinematics.forward_kinematics(q0)
        start_pos = T_start[:3, 3]
        start_R   = T_start[:3, :3]

        # Re-enable torque, then immediately command current position as goal (prevents snap-back)
        robot.bus.enable_torque()
        robot.send_action(robot_obs0)

        offset = start_pos - ep_ref_pos
        print(f"Start pos: {start_pos.round(4)}  offset from ref: {offset.round(4)}")
        print("Replaying episode in delta mode. Press Ctrl-C to abort.")

        for step in range(n_frames):
            t0 = time.perf_counter()

            global_idx = int(global_indices[step])
            pose = eef_poses[global_idx]
            ep_pos = pose[:3]
            ep_R   = pose[3:].reshape(3, 3)

            # Position delta from episode start, applied to robot start
            delta_pos = ep_pos - ep_ref_pos
            target_pos = start_pos + delta_pos

            # Keep orientation fixed to where user placed the arm
            rotvec = Rotation.from_matrix(start_R).as_rotvec()

            ee_action = {
                "ee.x": float(target_pos[0]),
                "ee.y": float(target_pos[1]),
                "ee.z": float(target_pos[2]),
                "ee.wx": float(rotvec[0]),
                "ee.wy": float(rotvec[1]),
                "ee.wz": float(rotvec[2]),
                "ee.gripper_pos": float(gripper_values[step]),
            }

            robot_obs = robot.get_observation()
            joint_action = ik_pipeline((ee_action, robot_obs))
            robot.send_action(joint_action)

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

        print("Replay complete.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
