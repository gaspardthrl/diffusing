#!/usr/bin/env python
"""Verify A5 FK/IK conventions: Pinocchio (training sidecars) vs Placo (deploy IK).

Run:
  uv run python scripts/verify_a5_kinematics.py
  uv run python scripts/verify_a5_kinematics.py --urdf ./so101_new_calib.urdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = REPO_ROOT / "so101_new_calib.urdf"
EEF_FRAME_ID = 15  # gripper_frame_link — must match precompute_eef_sidecars.py


def pinocchio_fk(joint_deg: np.ndarray, urdf_path: Path) -> np.ndarray:
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    q = np.zeros(model.nq)
    q[:5] = np.deg2rad(joint_deg[:5])
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[EEF_FRAME_ID]
    T = np.eye(4)
    T[:3, :3] = oMf.rotation
    T[:3, 3] = oMf.translation
    return T


def placo_fk(joint_deg: np.ndarray, urdf_path: Path, joint_names: list[str]) -> np.ndarray:
    from lerobot.model.kinematics import RobotKinematics

    kin = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=joint_names,
    )
    return kin.forward_kinematics(joint_deg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    args = ap.parse_args()
    if not args.urdf.exists():
        raise SystemExit(f"URDF not found: {args.urdf}")

    # Typical SO-101 pose (degrees): 5 arm + gripper
    joint_deg = np.array([0.0, -30.0, 45.0, 20.0, 0.0, 50.0], dtype=np.float64)
    motor_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    T_pin = pinocchio_fk(joint_deg, args.urdf)
    T_placo_all = placo_fk(joint_deg, args.urdf, motor_names)
    T_placo_arm = placo_fk(joint_deg[:5], args.urdf, motor_names[:5])

    def report(label: str, T: np.ndarray) -> None:
        print(f"{label}:")
        print(f"  pos = {T[:3, 3]}")
        print(f"  |pos| = {np.linalg.norm(T[:3, 3]):.4f} m")

    print(f"URDF: {args.urdf}\n")
    report("Pinocchio FK (training, 5 arm joints)", T_pin)
    report("Placo FK (all 6 motors)", T_placo_all)
    report("Placo FK (5 arm motors only)", T_placo_arm)

    err_all = np.linalg.norm(T_pin[:3, 3] - T_placo_all[:3, 3])
    err_arm = np.linalg.norm(T_pin[:3, 3] - T_placo_arm[:3, 3])
    R_err = np.linalg.norm(T_pin[:3, :3] - T_placo_arm[:3, :3], ord="fro")
    print(f"\nPosition error Pin vs Placo(6): {err_all * 1000:.2f} mm")
    print(f"Position error Pin vs Placo(5): {err_arm * 1000:.2f} mm")
    print(f"Rotation Frobenius error Pin vs Placo(5): {R_err:.4f}")

    if err_arm * 1000 > 5.0:
        print(
            "\nWARN: Pinocchio (A5 training poses) and Placo FK disagree by >5 mm.\n"
            "deploy_eef.py must use Pinocchio for T_ref (chunk anchor), not Placo."
        )
    else:
        print("\nOK: Pinocchio and Placo(5) FK agree within 5 mm — shared URDF/frame is consistent.")


if __name__ == "__main__":
    main()
