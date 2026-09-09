# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pins the gripperhead FDM/IDM recipe to ONE source of truth: ``GripperheadConfig``.

The recipe is spelled out in three places for three different readers — the pydantic schema
(``GripperheadConfig``), the dataset factory signature (``get_gripperhead_fdm_sft_dataset``) and
the shipped TOML (``examples/toml/sft_config/gripperhead_fdm_nano.toml``). These tests fail if any
of them drifts, which is how the repo previously ended up shipping "legacy" defaults that no run
actually used.

They also guard the removal of the ``GRIPPERHEAD_*`` environment-variable layer: the recipe must
be reachable only through the TOML + ``-- key=value`` CLI overrides.
"""

from __future__ import annotations

import inspect
import re
import tomllib
from pathlib import Path

import pytest

from cosmos_framework.configs.toml_config.sft_config import (
    GripperheadConfig,
    gripperhead_derived_overrides,
)
from cosmos_framework.configs.toml_config.toml_config_helper import PATH_REMAPS, _apply_remap
from cosmos_framework.data.vfm.action.datasets.gripperhead_fdm_dataset import (
    get_gripperhead_fdm_sft_dataset,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIPPED_TOML = _REPO_ROOT / "examples/toml/sft_config/gripperhead_fdm_nano.toml"
_DATASET_NODE = ("dataloader_train", "dataloader", "datasets", "gripperhead", "dataset")


def _dataset_field_for(config_field: str) -> str | None:
    """The dataset kwarg a ``[gripperhead]`` field is routed to, or None if it goes elsewhere.

    Resolved through ``PATH_REMAPS`` so the routing table itself is under test.
    """
    path = _apply_remap(PATH_REMAPS["vfm"], ["gripperhead", config_field])
    if path is None or tuple(path[:-1]) != _DATASET_NODE:
        return None
    return path[-1]


@pytest.mark.L0
def test_every_gripperhead_field_is_routed_somewhere() -> None:
    """A new [gripperhead] field must get an explicit home: the dataset node, another node, or an
    explicit ``None`` (derived). Otherwise it silently becomes a dead knob."""
    for field in GripperheadConfig.model_fields:
        path = _apply_remap(PATH_REMAPS["vfm"], ["gripperhead", field])
        # ``_apply_remap`` returns the input unchanged when NO rule matched. The catch-all
        # ("gripperhead",) rule means that can only happen if the catch-all is removed.
        assert path != ["gripperhead", field], f"[gripperhead].{field} has no routing rule"


@pytest.mark.L0
def test_schema_defaults_match_dataset_defaults() -> None:
    """``GripperheadConfig`` and the dataset factory must agree on every shared default.

    Both are user-visible entry points, so a mismatch means the recipe you get depends on which
    door you came through.
    """
    signature = inspect.signature(get_gripperhead_fdm_sft_dataset).parameters
    recipe = GripperheadConfig()
    checked = []
    for field in GripperheadConfig.model_fields:
        dataset_field = _dataset_field_for(field)
        if dataset_field is None:
            continue  # routed to the model/checkpoint instead — covered by the derived tests
        assert dataset_field in signature, (
            f"[gripperhead].{field} routes to {dataset_field!r}, which is not a "
            f"get_gripperhead_fdm_sft_dataset parameter"
        )
        expected = getattr(recipe, field)
        actual = signature[dataset_field].default
        if field == "roots":
            continue  # environment-specific path, intentionally empty in both
        assert actual == expected, (
            f"default drift: GripperheadConfig.{field}={expected!r} but "
            f"get_gripperhead_fdm_sft_dataset({dataset_field}=...)={actual!r}"
        )
        checked.append(field)
    assert len(checked) > 15, f"expected to check most of the recipe, only compared {checked}"


@pytest.mark.L0
def test_shipped_toml_matches_schema_defaults() -> None:
    """The shipped TOML spells out the recipe for readability; it must not disagree with it."""
    with open(_SHIPPED_TOML, "rb") as fh:
        block = tomllib.load(fh)["gripperhead"]
    recipe = GripperheadConfig()
    for key, value in block.items():
        assert key in GripperheadConfig.model_fields, f"{_SHIPPED_TOML.name}: unknown key {key!r}"
        if isinstance(value, str) and value.startswith("${"):
            continue  # env interpolation (paths)
        assert value == getattr(recipe, key), (
            f"{_SHIPPED_TOML.name}: [gripperhead].{key}={value!r} disagrees with "
            f"GripperheadConfig.{key}={getattr(recipe, key)!r}"
        )


@pytest.mark.L0
def test_derived_overrides_track_the_horizon() -> None:
    """``encode_exact_durations`` is the coupled value that used to be computed at import time from
    the environment. It must follow num_pred / num_history / calib_seg_len."""
    def durations(**kwargs) -> str:
        overrides = gripperhead_derived_overrides(
            GripperheadConfig.model_validate({**GripperheadConfig().model_dump(), **kwargs})
        )
        (line,) = [o for o in overrides if "encode_exact_durations" in o]
        return line.split("=", 1)[1]

    # Shipped recipe: calib segment 5, current+future 16+1, sparse history 25.
    assert durations() == "[5,17,25]"
    # A CLI override of the history length must move with it.
    assert durations(num_history=9) == "[5,9,17]"
    # num_history=1 => no separate history item.
    assert durations(num_history=1) == "[5,17]"
    # Calibration off => no calib-segment length.
    assert durations(use_calibration=False) == "[17,25]"
    # A longer prediction horizon.
    assert durations(num_pred=32) == "[5,25,33]"


@pytest.mark.L0
def test_derived_overrides_gate_axis_aug_and_action_heads() -> None:
    base = GripperheadConfig()
    assert base.p_pose_axis_aug > 0
    assert any(o.endswith(".axis_aug_enabled=true") for o in gripperhead_derived_overrides(base))

    off = GripperheadConfig(p_pose_axis_aug=0.0)
    assert any(o.endswith(".axis_aug_enabled=false") for o in gripperhead_derived_overrides(off))

    # resume_action_heads=True inherits the trained heads; False re-initializes them.
    (inherit,) = [o for o in gripperhead_derived_overrides(base) if "keys_to_skip_loading" in o]
    assert inherit == "checkpoint.keys_to_skip_loading=[net_ema.]"
    fresh_cfg = GripperheadConfig(resume_action_heads=False)
    (fresh,) = [o for o in gripperhead_derived_overrides(fresh_cfg) if "keys_to_skip_loading" in o]
    assert "action2llm" in fresh and "llm2action" in fresh


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_pred": 15}, "multiple of 4"),
        ({"num_pred": 0}, "multiple of 4"),
        ({"num_history": 10}, r"4n\+1"),
    ],
)
@pytest.mark.L0
def test_invalid_horizons_are_rejected_up_front(kwargs, message) -> None:
    """A bad horizon must fail at config time, not at the first VAE encode."""
    cfg = GripperheadConfig.model_validate({**GripperheadConfig().model_dump(), **kwargs})
    with pytest.raises(ValueError, match=message):
        gripperhead_derived_overrides(cfg)


@pytest.mark.L0
def test_no_gripperhead_env_vars_in_the_training_path() -> None:
    """The ``GRIPPERHEAD_*`` recipe env layer is gone; only documented PATH env vars remain.

    ``GRIPPERHEAD_DATA_ROOTS`` is allowed because it is a filesystem path supplied through the
    TOML's ``${oc.env:...}`` interpolation, like BASE_CHECKPOINT_PATH and WAN_VAE_PATH.
    """
    allowed = {"GRIPPERHEAD_DATA_ROOTS"}
    tracked = [
        _REPO_ROOT / "cosmos_framework/configs/base/experiment/action/posttrain_config/gripperhead_fdm_nano.py",
        _REPO_ROOT / "cosmos_framework/data/vfm/action/datasets/gripperhead_fdm_dataset.py",
        _REPO_ROOT / "cosmos_framework/configs/toml_config/sft_config.py",
        _REPO_ROOT / "cosmos_framework/configs/toml_config/toml_config_helper.py",
        _SHIPPED_TOML,
        # The eval script never read these — it only carried "(GRIPPERHEAD_X)" annotations pointing
        # at the launcher's env plumbing, which went stale the moment the layer was removed. Keep it
        # in the guard so cross-references stay pointed at real [gripperhead] fields.
        _REPO_ROOT / "examples/eval_gripperhead_fdm_rollout.py",
    ]
    offenders: dict[str, set[str]] = {}
    for path in tracked:
        found = set(re.findall(r"GRIPPERHEAD_[A-Z0-9_]+", path.read_text())) - allowed
        if found:
            offenders[path.name] = found
    assert not offenders, f"GRIPPERHEAD_* env vars reintroduced: {offenders}"


@pytest.mark.L0
def test_experiment_module_reads_no_environment() -> None:
    """The experiment config must be a pure function of the TOML + CLI, so a run is reproducible
    from its recorded config alone."""
    source = (
        _REPO_ROOT / "cosmos_framework/configs/base/experiment/action/posttrain_config/gripperhead_fdm_nano.py"
    ).read_text()
    assert "os.environ" not in source
    assert "getenv" not in source
