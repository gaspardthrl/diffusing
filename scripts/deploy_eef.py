"""Deploy an EEF-delta diffusion policy (A5) on the SO-101 arm.

Training uses Pinocchio FK sidecars (eef_poses.npy). Inference must:
  T_ref = Pinocchio FK at chunk start (5 arm joints)
  T_target = T_ref + policy delta (matches EEFActionProcessorStep)
  joints = Pinocchio IK(T_target); gripper is absolute from policy

lerobot-rollout / scripts/deploy.sh do NOT run IK — use this script for A5.

Usage:
  uv run python scripts/deploy_eef.py --repo gaspardthrl/walleed-diffusion-A5
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "lerobot" / "src"))

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.model.pinocchio_kinematics import PinocchioKinematics
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor.eef_action_processor import EEFUnnormalizeProcessorStep
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.gripper_orientation import constrain_gripper_approach
from lerobot.utils.robot_utils import precise_sleep

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "so101_new_calib.urdf"
DEFAULT_CALIBRATION_DIR = REPO_ROOT / "calibration"
ENV_PATH = REPO_ROOT / ".env"
URDF_SOURCES = [
    ("TheRobotStudio/SO-ARM100",
     "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf"),
    ("HuggingFace haixuantao/dora-bambot",
     "https://huggingface.co/haixuantao/dora-bambot/resolve/main/URDF/so101.urdf"),
]
PINOCCHIO_EEF_FRAME_ID = 15


def _load_env() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


def _ensure_urdf(path: Path) -> Path:
    if path.is_file():
        return path
    for name, url in URDF_SOURCES:
        try:
            print(f"URDF not found; downloading from {name} ...")
            urllib.request.urlretrieve(url, path)
            print(f"URDF saved to {path}")
            return path
        except Exception as e:
            print(f"URDF download from {name} failed ({e}), trying next ...")
    raise FileNotFoundError(
        f"URDF not found: {path}. Place so101_new_calib.urdf in repo root "
        "or pass --urdf (same file as precompute_eef_sidecars.py)."
    )


def sixd_to_rotmat(v: np.ndarray) -> np.ndarray:
    a1, a2 = v[:3], v[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def compute_target_pose(T_ref: np.ndarray, pos_delta: np.ndarray, rot_6d: np.ndarray) -> np.ndarray:
    T_target = np.eye(4, dtype=np.float64)
    R_delta = sixd_to_rotmat(rot_6d.astype(np.float64))
    T_target[:3, 3] = T_ref[:3, 3] + pos_delta.astype(np.float64)
    T_target[:3, :3] = T_ref[:3, :3] @ R_delta
    return T_target


def run(args: argparse.Namespace) -> None:
    _load_env()

    camera_config = {
        "front": OpenCVCameraConfig(
            index_or_path=args.camera_index,
            width=640, height=480, fps=args.fps,
        )
    }
    robot_config = SO101FollowerConfig(
        port=args.port,
        id=args.robot_id,
        cameras=camera_config,
        calibration_dir=args.calibration_dir,
        use_degrees=True,
    )
    robot = SO101Follower(robot_config)

    print(f"Loading policy {args.repo} @ {args.revision} ...")
    ckpt_path = snapshot_download(repo_id=args.repo, revision=args.revision, repo_type="model")
    policy = DiffusionPolicy.from_pretrained(ckpt_path)
    device = torch.device(policy.config.device)
    policy = policy.to(device).eval()
    policy.reset()
    print(f"  device={device}")
    n_action_steps = args.n_action_steps or policy.config.n_action_steps
    print(f"  n_action_steps={n_action_steps}  horizon={policy.config.horizon}")

    print(f"Loading EEF stats from {args.eef_stats} ...")
    unnorm = EEFUnnormalizeProcessorStep(args.eef_stats)

    urdf_path = _ensure_urdf(Path(args.urdf))
    print(f"Loading Pinocchio kinematics from {urdf_path} ...")
    kin = PinocchioKinematics(urdf_path, frame_id=PINOCCHIO_EEF_FRAME_ID)
    robot.connect()
    motor_names = list(robot.bus.motors.keys())

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt_path,
        preprocessor_overrides={
            "device_processor": {"device": policy.config.device},
        },
    )

    MOTOR_KEYS = [f"{m}.pos" for m in motor_names]
    obs_hw = {
        k: v
        for k, v in robot.observation_features.items()
        if k.endswith(".pos") or isinstance(v, tuple)
    }
    ds_features = hw_to_dataset_features(obs_hw, OBS_STR, use_video=False)

    print("Starting control loop. Press Ctrl+C to stop.")
    print(f"  Robot id: {args.robot_id}  port: {args.port}")
    print(f"  Calibration: {args.calibration_dir / f'{args.robot_id}.json'}")
    print(f"  FK/IK: Pinocchio frame id={PINOCCHIO_EEF_FRAME_ID} ({urdf_path.name})")
    if args.constrain_no_forward:
        print(f"  EE cone: approach col {args.approach_axis} > {args.max_forward_angle_deg:.0f}° from forward")
    control_interval = 1.0 / args.fps

    try:
        while True:
            obs_raw = robot.get_observation()
            joint_angles = np.array([obs_raw[k] for k in MOTOR_KEYS], dtype=np.float64)
            T_ref = kin.fk(joint_angles)

            obs_frame = build_dataset_frame(ds_features, obs_raw, prefix=OBS_STR)
            if not policy.config.state_indices:
                obs_frame["observation.state"] = np.zeros(6, dtype=np.float32)
            batch = prepare_observation_for_inference(
                obs_frame, device, task=None, robot_type=robot.robot_type
            )
            batch = preprocessor(batch)

            with torch.no_grad():
                norm_chunk = policy.predict_action_chunk(batch)
            norm_chunk = norm_chunk[:, :n_action_steps]

            transition = {TransitionKey.ACTION: norm_chunk}
            transition = unnorm(transition)
            chunk = transition[TransitionKey.ACTION][0].cpu().numpy()

            warm_joints = joint_angles.copy()
            for t in range(n_action_steps):
                step = chunk[t]
                pos_delta = step[:3]
                rot_6d = step[3:9]
                gripper = step[9]

                T_target = compute_target_pose(T_ref, pos_delta, rot_6d)
                if args.constrain_no_forward:
                    T_target[:3, :3] = constrain_gripper_approach(
                        T_target[:3, :3],
                        forward_axis=args.forward_axis,
                        max_forward_dot=args.max_forward_dot,
                        approach_axis=args.approach_axis,
                        down_axis=args.down_axis,
                    )

                target_joints = kin.inverse_kinematics(
                    warm_joints,
                    T_target,
                    position_weight=args.ik_pos_w,
                    orientation_weight=args.ik_ori_w,
                )
                target_joints[-1] = gripper
                warm_joints = target_joints.copy()

                action = {k: float(target_joints[i]) for i, k in enumerate(MOTOR_KEYS)}
                robot.send_action(action)
                precise_sleep(control_interval)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.disconnect()


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="gaspardthrl/walleed-diffusion-A5")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--eef-stats", default=str(REPO_ROOT / "data_combined/meta/eef_stats.json"))
    ap.add_argument("--urdf", default=str(DEFAULT_URDF))
    ap.add_argument("--port", default=os.environ.get("FOLLOWER_PORT", "/dev/tty.usbmodem5B141129431"))
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--robot-id", default=os.environ.get("FOLLOWER_ID", "walleed_follower"))
    ap.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--ik-pos-w", type=float, default=1.0)
    ap.add_argument("--ik-ori-w", type=float, default=0.01)
    ap.add_argument(
        "--constrain-no-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--max-forward-angle-deg", type=float, default=55.0)
    ap.add_argument("--forward-axis", type=float, nargs=3, default=(1.0, 0.0, 0.0))
    ap.add_argument("--down-axis", type=float, nargs=3, default=(0.0, 0.0, -1.0))
    ap.add_argument("--approach-axis", type=int, choices=(0, 1, 2), default=2)
    args = ap.parse_args()
    args.max_forward_dot = float(np.cos(np.deg2rad(args.max_forward_angle_deg)))
    args.forward_axis = np.array(args.forward_axis, dtype=np.float64)
    args.down_axis = np.array(args.down_axis, dtype=np.float64)
    run(args)


if __name__ == "__main__":
    main()
