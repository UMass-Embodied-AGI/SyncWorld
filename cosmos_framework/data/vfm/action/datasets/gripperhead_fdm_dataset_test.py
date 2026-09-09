# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import pickle
import random

import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as T

from cosmos_framework.data.vfm.action.action_spec import Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.datasets.gripperhead_fdm_dataset import (
    GripperheadFDMDataset,
    build_padded_sparse_history_indices,
    NEUTRAL_CAPTION_SEED,
    load_gripperhead_caption,
    sample_history_real_frames,
)


@pytest.mark.L0
def test_load_gripperhead_caption_prefers_leaf_metadata(tmp_path) -> None:
    leaf = tmp_path / "PickPlaceCounterToCabinet" / "episode_0" / "camera_poses_0" / "expert"
    leaf.mkdir(parents=True)
    with open(leaf.parent / "base_init.pkl", "wb") as file:
        pickle.dump({"lang": "legacy instruction"}, file)
    with open(leaf / "metadata.json", "w", encoding="utf-8") as file:
        json.dump({"lang": "put the mug in the cabinet"}, file)

    assert load_gripperhead_caption(str(leaf)) == "put the mug in the cabinet"


@pytest.mark.L0
def test_load_gripperhead_caption_reads_legacy_base_init(tmp_path) -> None:
    leaf = tmp_path / "OpenDrawer" / "episode_0" / "camera_poses_0" / "expert"
    leaf.mkdir(parents=True)
    with open(leaf.parent / "base_init.pkl", "wb") as file:
        pickle.dump({"lang": "open the left drawer"}, file)

    assert load_gripperhead_caption(str(leaf)) == "open the left drawer"


@pytest.mark.L0
def test_caption_falls_back_to_neutral_seed_not_directory_name(tmp_path) -> None:
    """No real ``lang`` anywhere => task-neutral seed, NOT the directory above ``episode_*``.

    Returning ``"pn p counter to cab"`` here would teach the text slot a scene/domain
    semantics (and leak episode identity for session-named sources like DROID).
    """
    leaf = tmp_path / "PnPCounterToCab" / "episode_42" / "expert"
    leaf.mkdir(parents=True)

    assert load_gripperhead_caption(str(leaf)) == NEUTRAL_CAPTION_SEED
    assert NEUTRAL_CAPTION_SEED != ""  # non-empty: the metadata appenders skip empty captions


@pytest.mark.L0
def test_caption_still_recovers_real_libero_instruction_from_path(tmp_path) -> None:
    """LIBERO filenames genuinely encode the instruction, so that path branch stays."""
    leaf = tmp_path / "libero_10__KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo" / "episode_0" / "expert"
    leaf.mkdir(parents=True)

    assert load_gripperhead_caption(str(leaf)) == "put the moka pot on the stove"


@pytest.mark.L0
def test_training_style_history_uses_sparse_frames_then_tail_padding() -> None:
    rng = random.Random(4)
    num_real = sample_history_real_frames(rng, minimum=2, maximum=5, available=3)
    indices = build_padded_sparse_history_indices(
        anchor=20,
        num_history_frames=5,
        history_frame_stride=3,
        num_real_frames=num_real,
    )

    assert 2 <= num_real <= 3
    assert indices[:num_real] == list(range(20 - 3 * num_real, 20, 3))
    assert indices[num_real:] == [indices[num_real - 1]] * (5 - num_real)


@pytest.mark.L0
def test_eval_future_length_is_never_randomly_truncated() -> None:
    """Training may sample a shorter future (>= future_min_frames); evaluation never does."""
    dataset = object.__new__(GripperheadFDMDataset)
    dataset.num_pred_frames = 16
    dataset.future_min_frames = 5
    dataset.rng = random.Random(0)

    dataset.is_train = False
    assert dataset._sample_future_length(mode="forward_dynamics", available=16) == 16

    dataset.is_train = True
    assert 5 <= dataset._sample_future_length(mode="inverse_dynamics", available=16) <= 16


@pytest.mark.parametrize("task_mode", ["forward_dynamics", "inverse_dynamics", "joint"])
@pytest.mark.L0
def test_fdm_idm_pretraining_always_uses_neutral_caption(tmp_path, task_mode) -> None:
    """FDM/IDM pretraining is instruction-free by design: a REAL ``lang`` on the leaf is ignored.

    All three task modes are pretraining modes, so this holds unconditionally. Only a downstream
    subclass that clears ``_ALLOW_NEUTRAL_CAPTION`` resolves the real instruction.
    """
    leaf = tmp_path / "OpenDrawer" / "episode_0" / "expert"
    leaf.mkdir(parents=True)
    with open(leaf / "metadata.json", "w", encoding="utf-8") as file:
        json.dump({"lang": "open the left drawer"}, file)

    dataset = object.__new__(GripperheadFDMDataset)
    dataset.task_mode = task_mode
    dataset.force_neutral_caption = GripperheadFDMDataset._ALLOW_NEUTRAL_CAPTION
    dataset._caption_cache = {}

    assert dataset._caption_for_leaf(str(leaf)) == NEUTRAL_CAPTION_SEED


@pytest.mark.L0
def test_neutral_caption_is_opt_out_only_via_class_flag(tmp_path) -> None:
    """A subclass that needs real task language clears ``_ALLOW_NEUTRAL_CAPTION``; nothing else
    (no task mode, no env var) can turn the instruction back on."""
    leaf = tmp_path / "OpenDrawer" / "episode_0" / "expert"
    leaf.mkdir(parents=True)
    with open(leaf / "metadata.json", "w", encoding="utf-8") as file:
        json.dump({"lang": "open the left drawer"}, file)

    assert GripperheadFDMDataset._ALLOW_NEUTRAL_CAPTION is True
    dataset = object.__new__(GripperheadFDMDataset)
    dataset.task_mode = "forward_dynamics"
    dataset.force_neutral_caption = False  # what a downstream opt-out produces
    dataset._caption_cache = {}

    assert dataset._caption_for_leaf(str(leaf)) == "open the left drawer"


def _make_hist_multiitem_dataset(p_mask_history: float = 0.0) -> GripperheadFDMDataset:
    """A bare hist>1, nocalib GripperheadFDMDataset wired for the _load_sample_multiitem history
    seam (in-memory arrays, no disk / mp4 decode)."""
    ds = object.__new__(GripperheadFDMDataset)
    ds.num_history_frames, ds.num_pred_frames, ds.history_frame_stride = 5, 4, 3
    ds.history_min_frames, ds.future_min_frames = 3, 2
    ds.fps = 15.0
    ds.is_train = True
    ds.use_wrist = ds.use_calibration = False
    ds.calib_num_segments = 6
    ds.contact_bias_prob = ds.p_null_action = 0.0
    ds.p_include_history = ds.p_include_calibration = 1.0
    ds.axis_aug_enabled = False
    ds.p_pose_axis_aug = 0.0
    ds.action_rotation_format = "rot6d"
    ds.action_pose_convention = "backward_framewise"
    ds.action_translation_scale = ds.action_rotation_scale = 1.0
    ds.domain_id = 0
    ds.viewpoint = "third_person_view"
    ds.force_neutral_caption = True  # fwd/inv pretraining: instruction-free by design
    ds.p_mask_history = float(p_mask_history)
    ds._caption_cache, ds._contact_cache = {}, {}
    ds._to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    ds._spec = build_action_spec(Pos(), Rot("rot6d"), Gripper())
    ds.rng = random.Random(0)
    return ds


def _hist_mask_synthetic():
    """Synthetic in-memory episode arrays with real (moving) history."""
    total = 40
    mats = np.stack([np.eye(4, dtype=np.float32) for _ in range(total)])
    mats[:, 0, 3] = 0.01 * np.arange(total, dtype=np.float32)
    gopen = np.ones(total, dtype=np.float32)
    rgb = np.zeros((total, 8, 8, 3), dtype=np.uint8)
    for t in range(total):
        rgb[t] = min(255, t * 6)
    epi = {"path": "/tmp/gh/PickPlace/episode_0/expert", "episode": "expert"}
    return epi, mats, gopen, rgb, total


@pytest.mark.L0
def test_multiitem_history_carries_real_motion() -> None:
    """hist>1 + nocalib => items are [history, current+future]; the history item holds REAL frames
    (not anchor-padding) and a moving history action."""
    epi, mats, gopen, rgb, total = _hist_mask_synthetic()
    ds = _make_hist_multiitem_dataset()
    out = ds._load_sample_multiitem(epi["path"], epi, mats, gopen, rgb, total, "forward_dynamics")

    hist_vid, cf_vid = out["video"]
    hist_act = out["action"][0]
    assert not torch.equal(hist_vid[:, 0], cf_vid[:, 0])
    assert not torch.allclose(hist_act[:, :3], torch.zeros_like(hist_act[:, :3]), atol=1e-6)


@pytest.mark.L0
def test_history_mask_off_emits_no_override() -> None:
    """p_mask_history=0.0 => no override is emitted: the sample's ``history_mask`` is None and the
    plan/packer are untouched downstream."""
    epi, mats, gopen, rgb, total = _hist_mask_synthetic()
    ds = _make_hist_multiitem_dataset(p_mask_history=0.0)
    out = ds._load_sample_multiitem(epi["path"], epi, mats, gopen, rgb, total, "forward_dynamics")
    assert out["history_mask"] is None


@pytest.mark.L0
def test_history_mask_forward_masks_history_video_latents() -> None:
    """forward_dynamics + p_mask_history=1.0 masks a recent contiguous run of REAL history VIDEO
    latents (target modality = video); no action steps are masked. nocalib => history item index 0."""
    epi, mats, gopen, rgb, total = _hist_mask_synthetic()
    ds = _make_hist_multiitem_dataset(p_mask_history=1.0)
    out = ds._load_sample_multiitem(epi["path"], epi, mats, gopen, rgb, total, "forward_dynamics")
    hm = out["history_mask"]
    assert hm is not None
    assert hm["item_index"] == 0  # nocalib => history is the first item
    assert hm["action_steps"] == []
    vf = hm["vision_frames"]
    n_lat = (ds.num_history_frames - 1) // 4 + 1  # total history latents
    assert len(vf) >= 1
    # Recent-end contiguous run of REAL latents: ends at the last real latent, indices in-range & contiguous.
    assert max(vf) <= n_lat - 1
    assert vf == list(range(vf[0], vf[-1] + 1))


@pytest.mark.L0
def test_history_mask_inverse_masks_history_action_steps() -> None:
    """inverse_dynamics + p_mask_history=1.0 masks a recent contiguous run of REAL history ACTION
    steps (target modality = action); no video latents are masked."""
    epi, mats, gopen, rgb, total = _hist_mask_synthetic()
    ds = _make_hist_multiitem_dataset(p_mask_history=1.0)
    out = ds._load_sample_multiitem(epi["path"], epi, mats, gopen, rgb, total, "inverse_dynamics")
    hm = out["history_mask"]
    assert hm is not None
    assert hm["item_index"] == 0
    assert hm["vision_frames"] == []
    steps = hm["action_steps"]
    hist_action_len = ds.num_history_frames - 1  # (H-1) history action steps
    assert len(steps) >= 1
    assert max(steps) <= hist_action_len - 1
    assert steps == list(range(steps[0], steps[-1] + 1))


@pytest.mark.L0
def test_history_mask_skipped_when_no_real_history() -> None:
    """k_hist == 0 leaves nothing real to mask => no override even at p=1.0."""
    epi, mats, gopen, rgb, total = _hist_mask_synthetic()
    ds = _make_hist_multiitem_dataset(p_mask_history=1.0)
    assert ds._pick_history_mask(k_hist=0, mode="forward_dynamics") is None
