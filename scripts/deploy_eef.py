"""Deploy an EEF-delta diffusion policy (A5) on the SO-101 robot.

Follows the same pattern as examples/so100_to_so100_EE/evaluate.py but adapted
for A5: the policy outputs 10-D EEF *delta* actions, not absolute EE poses.
We therefore apply FK once per action chunk to get the reference EEF pose, then
run IK for each of the n_action_steps steps in the chunk.

Usage:
  python scripts/deploy_eef.py \
      --repo gaspardthrl/walleed-diffusion-A5 \
      --revision step-40000 \
      --eef-stats outputs/eef_sidecars/eef_stats.json \
      --urdf ./SO101/so101_new_calib.urdf \
      --port /dev/tty.usbmodem5A460814411 \
      --camera-index 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "lerobot" / "src"))

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.processor.eef_action_processor import EEFUnnormalizeProcessorStep
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.types import TransitionKey
from lerobot.utils.robot_utils import precise_sleep


# ── Rotation helpers ──────────────────────────────────────────────────────────

def sixd_to_rotmat(v: np.ndarray) -> np.ndarray:
    """Gram-Schmidt from 6D [col0(3), col1(3)] → (3,3)."""
    a1, a2 = v[:3], v[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)   # columns


def compute_target_pose(T_ref: np.ndarray, pos_delta: np.ndarray, rot_6d: np.ndarray) -> np.ndarray:
    """Reconstruct absolute target EEF pose from chunk-start reference + delta.

    Training convention (EEFActionProcessorStep):
        delta_pos = pos[k] - pos_ref          →  pos_target = pos_ref + delta_pos
        R_delta   = R_ref.T @ R[k]            →  R_target   = R_ref  @ R_delta
    """
    T_target = np.eye(4, dtype=np.float64)
    R_delta = sixd_to_rotmat(rot_6d.astype(np.float64))
    T_target[:3, 3]  = T_ref[:3, 3] + pos_delta.astype(np.float64)
    T_target[:3, :3] = T_ref[:3, :3] @ R_delta
    return T_target


# ── Control loop ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Robot ─────────────────────────────────────────────────────────────────
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
        use_degrees=True,
    )
    robot = SO101Follower(robot_config)

    # ── Policy ────────────────────────────────────────────────────────────────
    print(f"Loading policy {args.repo} @ {args.revision} ...")
    ckpt_path = snapshot_download(repo_id=args.repo, revision=args.revision, repo_type="model")
    policy = DiffusionPolicy.from_pretrained(ckpt_path).to(device).eval()
    policy.reset()
    n_action_steps = args.n_action_steps or policy.config.n_action_steps
    print(f"  n_action_steps={n_action_steps}  horizon={policy.config.horizon}")

    # ── EEF unnormalization (replaces lerobot's default AbsoluteActionsProcessorStep) ──
    print(f"Loading EEF stats from {args.eef_stats} ...")
    unnorm = EEFUnnormalizeProcessorStep(args.eef_stats)

    # ── Kinematics ────────────────────────────────────────────────────────────
    print(f"Loading kinematics from {args.urdf} ...")
    # joint_names from robot.bus.motors to match motor ordering exactly
    robot.connect()
    motor_names = list(robot.bus.motors.keys())
    kin = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    # Preprocessor (handles ImageNet normalization, state normalization, etc.)
    # We build from the saved policy config + dataset stats (the policy already
    # has stats embedded via use_imagenet_stats; no separate dataset needed).
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt_path,
    )

    MOTOR_KEYS = [f"{m}.pos" for m in motor_names]   # order matches joint angles array
    IMAGE_KEY  = "observation.images.front"

    print("Starting control loop. Press Ctrl+C to stop.")
    control_interval = 1.0 / args.fps

    try:
        while True:
            # ── 1. Read observation ──────────────────────────────────────────
            obs = robot.get_observation()
            # Joint angles as ordered array (degrees)
            joint_angles = np.array([obs[k] for k in MOTOR_KEYS], dtype=np.float64)
            # Camera image: numpy (H, W, 3) uint8, returned under the full key
            img = obs[IMAGE_KEY]   # shape (480, 640, 3) uint8

            # ── 2. FK reference (captured ONCE per chunk) ────────────────────
            T_ref = kin.forward_kinematics(joint_angles)   # (4,4)

            # ── 3. Policy forward: produce normalized 10D EEF delta chunk ────
            # Build the policy input batch.
            # A5 is vision-only: state is passed as zeros (state_indices=[] at training).
            img_t = (
                torch.from_numpy(img)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )   # (1, 3, H, W) — encoder applies grayworld + resize + ImageNet norm internally

            state_t = torch.zeros(1, 6, device=device)   # (1, 6) zeroed proprio

            batch = {
                IMAGE_KEY:           img_t,
                "observation.state": state_t,
            }

            with torch.no_grad():
                # Returns (1, horizon, 10) normalized — full chunk
                norm_chunk = policy.predict_action_chunk(batch)
            norm_chunk = norm_chunk[:, :n_action_steps]   # (1, T, 10)

            # ── 4. Un-normalise EEF deltas ────────────────────────────────────
            transition = {TransitionKey.ACTION: norm_chunk}
            transition = unnorm(transition)
            chunk = transition[TransitionKey.ACTION][0].cpu().numpy()   # (T, 10)

            # ── 5. Execute chunk step by step ─────────────────────────────────
            warm_joints = joint_angles.copy()   # IK warm-start
            for t in range(n_action_steps):
                step = chunk[t]
                pos_delta = step[:3]    # metres
                rot_6d    = step[3:9]   # 6D rotation delta
                gripper   = step[9]     # absolute gripper (degrees)

                T_target = compute_target_pose(T_ref, pos_delta, rot_6d)

                target_joints = kin.inverse_kinematics(
                    warm_joints,
                    T_target,
                    position_weight=args.ik_pos_w,
                    orientation_weight=args.ik_ori_w,
                )
                # Policy controls gripper absolutely; override IK's gripper output
                target_joints[-1] = gripper
                warm_joints = target_joints.copy()   # seed next IK step

                action = {k: float(target_joints[i]) for i, k in enumerate(MOTOR_KEYS)}
                robot.send_action(action)
                precise_sleep(control_interval)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.disconnect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo",       default="gaspardthrl/walleed-diffusion-A5")
    ap.add_argument("--revision",   default="step-20000",
                    help="HF branch/revision (e.g. step-40000, main)")
    ap.add_argument("--eef-stats",  default="outputs/eef_sidecars/eef_stats.json")
    ap.add_argument("--urdf",       default="./SO101/so101_new_calib.urdf")
    ap.add_argument("--port",       default="/dev/tty.usbmodem5A460814411",
                    help="Serial port for the SO-101 arm")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--robot-id",   default="so101_follower")
    ap.add_argument("--fps",        type=float, default=30.0)
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="Steps to execute per chunk (default: policy config value)")
    ap.add_argument("--ik-pos-w",   type=float, default=1.0)
    ap.add_argument("--ik-ori-w",   type=float, default=0.01,
                    help="Orientation weight for IK (default 0.01 matches robot_kinematic_processor.py)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
