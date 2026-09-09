# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cosmos_framework.data.vfm.action.pose_utils import pose_abs_to_rel
from cosmos_framework.data.vfm.action.robocasa_native_action import (
    ACTION_SEMANTICS,
    FORMAT_NAME,
    ROTATION_SCALE,
    TRANSLATION_SCALE,
    action7_to_body_delta,
    action7_to_native_osc_arm,
    body_delta_to_action7,
    integrate_action7,
    native_osc_action_to_action7,
    validate_actions_payload,
)


def _random_pose(rng: np.random.Generator) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    pose[:3, 3] = rng.normal(size=3)
    return pose


def _payload(actions: np.ndarray) -> dict:
    return {
        "format": FORMAT_NAME,
        "action_semantics": ACTION_SEMANTICS,
        "actions": actions,
        "num_video_frames": len(actions) + 1,
        "control_fps": 20.0,
        "save_stride": 1,
        "translation_unit": "centimeter",
        "rotation_representation": "euler_xyz",
        "rotation_unit": "degree",
        "gripper_open_is_one": True,
    }


def test_native_osc_action_round_trip() -> None:
    rng = np.random.default_rng(7)
    for _ in range(100):
        native = np.zeros(12, dtype=np.float32)
        native[:6] = rng.uniform(-0.9, 0.9, size=6)
        native[6] = rng.choice([-1.0, 1.0])
        native[11] = -1.0
        current = _random_pose(rng)
        action7 = native_osc_action_to_action7(native, current)
        reconstructed = action7_to_native_osc_arm(action7, current)
        np.testing.assert_allclose(reconstructed, native[:7], atol=2e-6, rtol=0)


def test_explicit_gripper_state_resolves_hold_command() -> None:
    native = np.zeros(12, dtype=np.float32)
    action7 = native_osc_action_to_action7(native, np.eye(4), gripper_open=0.0)
    assert action7[6] == 0.0


def test_action7_matrix_and_integrated_trajectory_round_trip() -> None:
    rng = np.random.default_rng(11)
    actions = np.concatenate(
        [
            rng.uniform(-2.0, 2.0, size=(20, 3)),
            rng.uniform(-15.0, 15.0, size=(20, 3)),
            rng.integers(0, 2, size=(20, 1)),
        ],
        axis=1,
    ).astype(np.float32)
    for action in actions:
        reconstructed = body_delta_to_action7(action7_to_body_delta(action), action[6])
        np.testing.assert_allclose(reconstructed, action, atol=2e-5, rtol=0)

    trajectory = integrate_action7(actions)
    recovered = pose_abs_to_rel(
        trajectory,
        rotation_format="euler_xyz",
        pose_convention="backward_framewise",
        translation_scale=TRANSLATION_SCALE,
        rotation_scale=ROTATION_SCALE,
    )
    np.testing.assert_allclose(recovered, actions[:, :6], atol=2e-4, rtol=0)


def test_payload_validation_enforces_alignment_and_semantics() -> None:
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 6] = 1.0
    normalized = validate_actions_payload(_payload(actions))
    assert normalized["actions"].shape == (4, 7)
    assert normalized["num_video_frames"] == 5

    malformed = _payload(actions)
    malformed["num_video_frames"] = 4
    with pytest.raises(ValueError, match=r"len\(actions\)\+1"):
        validate_actions_payload(malformed)

    malformed = _payload(actions)
    malformed["rotation_unit"] = "radian"
    with pytest.raises(ValueError, match="rotation_unit"):
        validate_actions_payload(malformed)
