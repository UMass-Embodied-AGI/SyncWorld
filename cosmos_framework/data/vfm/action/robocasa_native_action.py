# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboCasa native OSC action conversion and ``actions.pkl`` validation.

The policy action is a body-frame transform of the controller goal, encoded as
``[translation_cm(3), euler_xyz_degrees(3), gripper_open(1)]``.  Unlike a
finite difference of measured end-effector poses, this is derived from the
official OSC command and therefore is not contaminated by controller lag.

This module intentionally depends only on NumPy and SciPy.  The RoboCasa data
generation environment can import it without importing torch or Cosmos model
code.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

FORMAT_NAME = "robocasa_native_action_v1"
ACTION_SEMANTICS = "body_frame_osc_goal_delta"
TRANSLATION_SCALE = 100.0
ROTATION_SCALE = 180.0 / np.pi
OSC_OUT_MAX_POS = 0.05
OSC_OUT_MAX_ROT = 0.5


def _as_transform(value: np.ndarray, *, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains non-finite values")
    return transform


def control_site_base_pose_from_obs(obs: dict[str, Any]) -> np.ndarray:
    """Return the OSC control-site pose in the robot base frame.

    RoboSuite keeps legacy ``base_to_eef_quat`` for the end-effector *body*,
    while OSC controls ``ref_ori_mat`` from the end-effector *site*.  The site
    quaternion is therefore required for exact native-command round trips.
    """

    position_key = "robot0_base_to_eef_pos"
    site_quaternion_key = "robot0_base_to_eef_quat_site"
    if position_key not in obs or site_quaternion_key not in obs:
        missing = [key for key in (position_key, site_quaternion_key) if key not in obs]
        raise KeyError(f"observation is missing OSC control-site fields: {missing}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(obs[position_key], dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(np.asarray(obs[site_quaternion_key], dtype=np.float64)).as_matrix()
    return transform


def native_osc_action_to_body_delta(
    native_action: np.ndarray,
    current_control_site_base: np.ndarray,
    *,
    out_max_pos: float = OSC_OUT_MAX_POS,
    out_max_rot: float = OSC_OUT_MAX_ROT,
) -> np.ndarray:
    """Convert a normalized RoboCasa OSC command into a body-frame goal delta."""

    action = np.asarray(native_action, dtype=np.float64)
    if action.ndim != 1 or action.shape[0] < 7:
        raise ValueError(f"native_action must be a vector with at least 7 entries, got {action.shape}")
    current = _as_transform(current_control_site_base, name="current_control_site_base")

    goal = current.copy()
    goal[:3, 3] = current[:3, 3] + action[:3] * float(out_max_pos)
    delta_base_rotation = Rotation.from_rotvec(action[3:6] * float(out_max_rot)).as_matrix()
    goal[:3, :3] = delta_base_rotation @ current[:3, :3]
    return np.linalg.inv(current) @ goal


def body_delta_to_action7(delta_body: np.ndarray, gripper_open: float) -> np.ndarray:
    """Encode a body-frame transform as ``[cm, euler_xyz_deg, open]``."""

    delta = _as_transform(delta_body, name="delta_body")
    action = np.empty(7, dtype=np.float32)
    action[:3] = delta[:3, 3] * TRANSLATION_SCALE
    action[3:6] = Rotation.from_matrix(delta[:3, :3]).as_euler("xyz", degrees=True)
    action[6] = 1.0 if float(gripper_open) >= 0.5 else 0.0
    return action


def action7_to_body_delta(action7: np.ndarray) -> np.ndarray:
    """Decode ``[cm, euler_xyz_deg, open]`` into its body-frame transform."""

    action = np.asarray(action7, dtype=np.float64)
    if action.shape != (7,):
        raise ValueError(f"action7 must have shape (7,), got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("action7 contains non-finite values")
    delta = np.eye(4, dtype=np.float64)
    delta[:3, 3] = action[:3] / TRANSLATION_SCALE
    delta[:3, :3] = Rotation.from_euler("xyz", action[3:6], degrees=True).as_matrix()
    return delta


def native_osc_action_to_action7(
    native_action: np.ndarray,
    current_control_site_base: np.ndarray,
    *,
    gripper_open: float | None = None,
    out_max_pos: float = OSC_OUT_MAX_POS,
    out_max_rot: float = OSC_OUT_MAX_ROT,
) -> np.ndarray:
    """Convert one official RoboCasa action directly to the policy's 7D label."""

    action = np.asarray(native_action, dtype=np.float64)
    delta = native_osc_action_to_body_delta(
        action,
        current_control_site_base,
        out_max_pos=out_max_pos,
        out_max_rot=out_max_rot,
    )
    open_state = (1.0 if action[6] <= 0.0 else 0.0) if gripper_open is None else float(gripper_open)
    return body_delta_to_action7(delta, open_state)


def action7_to_native_osc_arm(
    action7: np.ndarray,
    current_control_site_base: np.ndarray,
    *,
    out_max_pos: float = OSC_OUT_MAX_POS,
    out_max_rot: float = OSC_OUT_MAX_ROT,
) -> np.ndarray:
    """Invert the conversion and return normalized ``[arm(6), gripper(1)]``."""

    current = _as_transform(current_control_site_base, name="current_control_site_base")
    action = np.asarray(action7, dtype=np.float64)
    goal = current @ action7_to_body_delta(action)
    delta_position_base = goal[:3, 3] - current[:3, 3]
    delta_rotation_base = goal[:3, :3] @ current[:3, :3].T

    native = np.empty(7, dtype=np.float32)
    native[:3] = np.clip(delta_position_base / float(out_max_pos), -1.0, 1.0)
    native[3:6] = np.clip(
        Rotation.from_matrix(delta_rotation_base).as_rotvec() / float(out_max_rot),
        -1.0,
        1.0,
    )
    native[6] = -1.0 if action[6] >= 0.5 else 1.0
    return native


def integrate_action7(actions: np.ndarray, initial_pose: np.ndarray | None = None) -> np.ndarray:
    """Integrate framewise body deltas into a command trajectory of length ``N+1``."""

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"actions must have shape (N, 7), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("actions contain non-finite values")
    current = np.eye(4, dtype=np.float64) if initial_pose is None else _as_transform(initial_pose, name="initial_pose")
    poses = [current.copy()]
    for action in values:
        current = current @ action7_to_body_delta(action)
        poses.append(current.copy())
    return np.asarray(poses, dtype=np.float64)


def validate_actions_payload(payload: Any, *, source: str = "actions.pkl") -> dict[str, Any]:
    """Validate and normalize the on-disk native-action payload."""

    if not isinstance(payload, dict):
        raise TypeError(f"{source}: payload must be a dict, got {type(payload).__name__}")
    if payload.get("format") != FORMAT_NAME:
        raise ValueError(f"{source}: expected format={FORMAT_NAME!r}, got {payload.get('format')!r}")
    if payload.get("action_semantics") != ACTION_SEMANTICS:
        raise ValueError(
            f"{source}: expected action_semantics={ACTION_SEMANTICS!r}, got {payload.get('action_semantics')!r}"
        )
    expected_metadata = {
        "translation_unit": "centimeter",
        "rotation_representation": "euler_xyz",
        "rotation_unit": "degree",
        "gripper_open_is_one": True,
        "save_stride": 1,
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise ValueError(f"{source}: expected {key}={expected!r}, got {payload.get(key)!r}")

    actions = np.asarray(payload.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{source}: actions must have shape (N, 7), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError(f"{source}: actions contain non-finite values")
    if len(actions) == 0:
        raise ValueError(f"{source}: actions is empty")
    if np.any((actions[:, 6] != 0.0) & (actions[:, 6] != 1.0)):
        raise ValueError(f"{source}: gripper labels must be binary with 1=open and 0=closed")

    num_video_frames = int(payload.get("num_video_frames", len(actions) + 1))
    if num_video_frames != len(actions) + 1:
        raise ValueError(
            f"{source}: num_video_frames must equal len(actions)+1, got {num_video_frames} vs {len(actions) + 1}"
        )
    control_fps = float(payload.get("control_fps", 0.0))
    if not np.isfinite(control_fps) or control_fps <= 0:
        raise ValueError(f"{source}: control_fps must be positive, got {control_fps}")

    normalized = dict(payload)
    normalized["actions"] = actions
    normalized["num_video_frames"] = num_video_frames
    normalized["control_fps"] = control_fps
    if "native_actions" in normalized:
        native = np.asarray(normalized["native_actions"], dtype=np.float32)
        if native.ndim != 2 or native.shape[0] != len(actions) or native.shape[1] < 7:
            raise ValueError(f"{source}: native_actions must have shape ({len(actions)}, D>=7), got {native.shape}")
        if not np.isfinite(native).all():
            raise ValueError(f"{source}: native_actions contain non-finite values")
        normalized["native_actions"] = native
    return normalized
