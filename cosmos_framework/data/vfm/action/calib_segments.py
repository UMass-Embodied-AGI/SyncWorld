# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Calibration axis-segmentation for gripperhead FDM.

Ported from robotics-wan-dev
``training/rlbench_dataset_calib_segments_context.py`` (``_rotmat_to_euler_zyx``,
``_mat4s_to_abs_pose6``, ``_axis_threshold``, ``_best_run_by_score``,
``_segment_indices_monotone_axis``, ``_build_calibration_axis_segments``,
``_pad_or_trim_segment_tensors``).

A "calibration" clip is the arm sweeping each DoF axis. We chop it into K per-axis
**segments** by detecting, for each (axis, sign), the best contiguous run of
monotone motion along that axis (thresholded by per-axis delta magnitude). This
module works purely on the calibration pose trajectory (T,4,4) and returns, per
segment, a list of ``seg_len`` frame indices into the calibration clip (pad/trim
by repeating the last index). The caller slices the calib video/poses by these
indices to build one conditioning vision item + action block per segment.

``efficient=True`` -> 6 segments (positive direction of each DoF):
    x_pos, y_pos, z_pos, yaw_pos, pitch_pos, roll_pos
``efficient=False`` -> 12 segments (both signs).

``seg_len`` must be ``1 + 4n`` (used as-is) or ``4n`` (built internally at
``seg_len + 1`` with an anchor step, first step dropped) so that each segment
VAE-encodes to a clean ``1 + (seg_len-1)/4`` latent-frame group.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# (name, dim, is_angle_deg, min_abs) — matches robotics-wan axis_specs
_AXIS_SPECS = [
    ("x", 0, False, 1e-3),
    ("y", 1, False, 1e-3),
    ("z", 2, False, 1e-3),
    ("yaw", 3, True, 0.2),
    ("pitch", 4, True, 0.2),
    ("roll", 5, True, 0.2),
]

EFFICIENT_SEGMENT_NAMES = ["x_pos", "y_pos", "z_pos", "yaw_pos", "pitch_pos", "roll_pos"]
FULL_SEGMENT_NAMES = [
    "x_pos", "x_neg", "y_pos", "y_neg", "z_pos", "z_neg",
    "yaw_pos", "yaw_neg", "pitch_pos", "pitch_neg", "roll_pos", "roll_neg",
]
# move_range.pkl movement_order tokens (e.g. "Z-", "Pitch+") -> axis dim in _AXIS_SPECS order.
_AXIS_NAME_TO_DIM = {"x": 0, "y": 1, "z": 2, "yaw": 3, "pitch": 4, "roll": 5}


def segment_names(efficient: bool) -> List[str]:
    return list(EFFICIENT_SEGMENT_NAMES if efficient else FULL_SEGMENT_NAMES)


def parse_movement_order(movement_order) -> dict:
    """Parse move_range.pkl `movement_order` (e.g. ['Z-','X+','Pitch+',...]) -> {dim: sign(+1/-1)}.

    The LIBERO/robosuite calibrations sweep each DoF in a per-episode sign (not always positive),
    so efficient mode uses these signs to detect the correct run per axis."""
    out: dict = {}
    for tok in (movement_order or []):
        t = str(tok).strip()
        if not t:
            continue
        sign = -1 if t.endswith("-") else 1
        name = t.rstrip("+-").lower()
        if name in _AXIS_NAME_TO_DIM:
            out[_AXIS_NAME_TO_DIM[name]] = sign
    return out


def _wrap_deg(a: np.ndarray) -> np.ndarray:
    return (a + 180.0) % 360.0 - 180.0


def _rotmat_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (yaw, pitch, roll) degrees, ZYX order (yaw=Z, pitch=Y, roll=X)."""
    R = np.asarray(R, dtype=np.float64)
    r00, r01 = float(R[0, 0]), float(R[0, 1])
    r10, r11 = float(R[1, 0]), float(R[1, 1])
    r20, r21, r22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    r20_clamped = max(-1.0, min(1.0, r20))
    pitch = np.arcsin(-r20_clamped)
    cp = np.cos(pitch)
    if cp < 1e-8:  # gimbal lock
        yaw = np.arctan2(-r01, r11) if r20 < 0 else np.arctan2(r01, r11)
        roll = 0.0
    else:
        yaw = np.arctan2(r10, r00)
        roll = np.arctan2(r21, r22)
    return np.array([_wrap_deg(np.degrees(yaw)), _wrap_deg(np.degrees(pitch)),
                     _wrap_deg(np.degrees(roll))], dtype=np.float64)


def _mat4s_to_abs_pose6(mats: np.ndarray) -> np.ndarray:
    """(T,4,4) abs mats -> (T,6) [x,y,z,yaw,pitch,roll] (angles in wrapped degrees)."""
    if mats.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)
    out = np.zeros((mats.shape[0], 6), dtype=np.float32)
    for i, m in enumerate(mats):
        out[i, :3] = m[:3, 3]
        out[i, 3:6] = _rotmat_to_euler_zyx(m[:3, :3])
    return out


def _axis_threshold(abs_deltas: np.ndarray, min_abs: float) -> float:
    if abs_deltas.size == 0:
        return float(min_abs)
    mx = float(abs_deltas.max())
    if mx <= 0.0:
        return float(min_abs)
    p90 = float(np.percentile(abs_deltas, 90))
    return max(float(min_abs), 0.2 * p90, 0.1 * mx)


def _best_run_by_score(deltas: np.ndarray, thr: float, sign: int) -> Tuple[int, int]:
    """Best contiguous run monotone in `sign`, maximizing sum(abs(delta)). (-1,-1) if none."""
    if deltas.size == 0:
        return -1, -1
    if sign >= 0:
        mask, score_vals = deltas > float(thr), deltas
    else:
        mask, score_vals = deltas < -float(thr), -deltas
    best_s = best_e = -1
    best_score = -1.0
    i, n = 0, int(deltas.shape[0])
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j, score = i, 0.0
        while j < n and bool(mask[j]):
            score += float(score_vals[j])
            j += 1
        run_len = j - i
        if (score > best_score) or (score == best_score and run_len > (best_e - best_s + 1)):
            best_score, best_s, best_e = score, i, j - 1
        i = j
    return best_s, best_e


def _segment_indices_monotone_axis(series: np.ndarray, is_angle_deg: bool, sign: int,
                                   min_abs: float, expand: int = 0) -> List[int]:
    """Frame indices (inclusive) of the best monotone run of `series` in direction `sign`."""
    T = int(series.shape[0])
    if T <= 0:
        return []
    if T == 1:
        return [0]
    if is_angle_deg:
        rad = np.unwrap(np.deg2rad(series.astype(np.float64)))
        deltas = np.rad2deg(np.diff(rad))
    else:
        deltas = np.diff(series.astype(np.float64))
    thr = _axis_threshold(np.abs(deltas), min_abs=float(min_abs))
    ds, de = _best_run_by_score(deltas, thr=thr, sign=int(sign))
    if ds < 0:
        return [0]
    frame_s, frame_e = ds, de + 1  # delta idx [ds..de] -> frame idx [ds..de+1]
    if expand > 0:
        frame_s = max(0, frame_s - int(expand))
        frame_e = min(T - 1, frame_e + int(expand))
    if frame_e < frame_s:
        return [0]
    return list(range(frame_s, frame_e + 1))


def _pad_or_trim_indices(idxs: List[int], target_len: int) -> List[int]:
    """Resize a run to target_len. If longer, EVENLY subsample across the whole run (so the
    segment spans the full axis sweep, not just its start); if shorter, pad by repeating the last."""
    if target_len <= 0:
        return []
    if len(idxs) == 0:
        return [0] * target_len
    if len(idxs) == target_len:
        return list(idxs)
    if len(idxs) > target_len:
        sel = np.linspace(0, len(idxs) - 1, target_len).round().astype(int)
        return [int(idxs[i]) for i in sel]
    return list(idxs) + [idxs[-1]] * (target_len - len(idxs))


def seg_len_internal(seg_len: int) -> Tuple[int, bool]:
    """Return (internal_len, drop_first). seg_len must be 1+4n (as-is) or 4n (+1 anchor, drop first)."""
    seg_len = int(seg_len)
    if seg_len % 4 == 1:
        return seg_len, False
    if seg_len % 4 == 0:
        return seg_len + 1, True
    raise ValueError(f"calib seg_len must be (1+4n) or (4n), got {seg_len}")


def _body_action_component(mats: np.ndarray, idxs: List[int], dim: int) -> float:
    """Signed magnitude of the BODY-frame action along ``dim`` accumulated over ``idxs``.

    This is the quantity the model actually consumes (the action block is built from
    backward_framewise body deltas ``T_i^-1 @ T_i+1``), so it — not the world-frame pose series —
    decides whether a calib segment reads as a POSITIVE sweep of that DoF."""
    if len(idxs) < 2:
        return 0.0
    tot = 0.0
    for a, b in zip(idxs[:-1], idxs[1:]):
        T = np.linalg.inv(np.asarray(mats[a], dtype=np.float64)) @ np.asarray(mats[b], dtype=np.float64)
        if dim < 3:
            tot += float(T[dim, 3])
        else:
            tot += float(_rotmat_to_euler_zyx(T[:3, :3])[dim - 3])   # (yaw,pitch,roll) = dims 3,4,5
    return tot


def _body_action_vec6(mats: np.ndarray, idxs: List[int]) -> np.ndarray:
    """Accumulated BODY-frame action over ``idxs`` as a 6-vector in _AXIS_SPECS order
    (x, y, z, yaw, pitch, roll). This is what the model consumes, so positivity must be judged here."""
    v = np.zeros(6, dtype=np.float64)
    if len(idxs) < 2:
        return v
    for a, b in zip(idxs[:-1], idxs[1:]):
        T = np.linalg.inv(np.asarray(mats[a], dtype=np.float64)) @ np.asarray(mats[b], dtype=np.float64)
        v[:3] += T[:3, 3]
        v[3:] += _rotmat_to_euler_zyx(T[:3, :3])          # (yaw, pitch, roll)
    return v


def build_calib_segment_indices(mats: np.ndarray, seg_len: int, efficient: bool = True,
                                movement_order=None, positive_body_actions: bool = True,
                                action_mats: "np.ndarray | None" = None) -> List[List[int]]:
    """Detect K per-axis calibration segments from calib poses.

    Args:
        mats: (T,4,4) absolute gripper poses of the calibration clip. Run DETECTION always uses these
            (raw, un-augmented) poses so the per-axis monotone-run detection + move_order signs stay valid.
        seg_len: OUTPUT frames per segment (1+4n or 4n).
        efficient: True -> 6 segments (one per DoF); False -> 12 (both signs per DoF).
        movement_order: optional move_range.pkl `movement_order`; in efficient mode its per-axis
            sign is used to detect the correct run (LIBERO calibs sweep some axes negative). If
            absent, efficient mode falls back to the positive direction per axis.
        positive_body_actions: efficient mode only. Pick, for each DoF slot, the run whose BODY-FRAME
            ACTION along that DoF is POSITIVE (the "<axis>_pos" contract), choosing among ALL 12
            candidate runs (6 axes x both sweep signs). Must be the same at training and
            evaluation time.
        action_mats: (T,4,4) poses used to EVALUATE the action for that choice — pass the
            AXIS-AUGMENTED poses when axis aug is active. Under the body-frame relabel R@S the action
            picks up S^T, which permutes AND flips components: the positive set BEFORE augmentation is
            not the positive set AFTER it. Selecting on the augmented action keeps every emitted slot
            positive in the frame the model actually sees. Defaults to ``mats`` (no augmentation).

    Returns:
        list of K index-lists, each of length `seg_len`, in canonical axis order (x,y,z,yaw,pitch,roll;
        both signs when not efficient), indexing frames of the calib clip (pad/trimmed; anchor dropped
        internally when seg_len is 4n).
    """
    internal_len, drop_first = seg_len_internal(seg_len)
    T = int(mats.shape[0])
    K = 6 if efficient else 12
    if T <= 0:
        return [[0] * seg_len for _ in range(K)]
    pose6 = _mat4s_to_abs_pose6(np.asarray(mats, dtype=np.float32))
    order_signs = parse_movement_order(movement_order) if efficient else {}

    def _emit(idxs: List[int]) -> List[int]:
        idxs = _pad_or_trim_indices(idxs or [0], internal_len)
        return idxs[1:] if drop_first else idxs   # drop the anchor step -> length seg_len

    if efficient and positive_body_actions:
        # Build ALL 12 candidate runs (6 axes x both sweep signs) from the RAW pose series, then score
        # each by its (possibly augmented) body-frame action and give slot j the run that moves most
        # POSITIVELY along DoF j. Scoring across all 12 — not just the two runs of axis j — is required
        # because S^T RELABELS axes: post-augmentation "+x" can be the pre-augmentation "-z" run.
        amats = np.asarray(mats if action_mats is None else action_mats, dtype=np.float64)
        cands: List[Tuple[List[int], np.ndarray]] = []
        for _n, d, ang, mabs in _AXIS_SPECS:
            for cand_sign in (+1, -1):
                c = _segment_indices_monotone_axis(pose6[:, int(d)], bool(ang), int(cand_sign),
                                                   float(mabs), expand=0)
                if c:
                    cands.append((c, _body_action_vec6(amats, c)))
        out: List[List[int]] = []
        for _n, d, _a, _m in _AXIS_SPECS:
            best, best_val = None, None
            for c, vec in cands:
                val = float(vec[int(d)])
                if best_val is None or val > best_val:
                    best, best_val = c, val
            out.append(_emit(list(best) if (best is not None and (best_val or 0.0) > 0.0) else [0]))
        assert len(out) == K and all(len(s) == seg_len for s in out), (
            f"expected {K} segments of len {seg_len}, got {[len(s) for s in out]}"
        )
        return out

    out: List[List[int]] = []
    for _name, dim, is_ang, min_abs in _AXIS_SPECS:
        series = pose6[:, int(dim)]
        signs = [int(order_signs.get(int(dim), 1))] if efficient else [+1, -1]
        for sign in signs:
            idxs = _segment_indices_monotone_axis(series, bool(is_ang), int(sign), float(min_abs), expand=0)
            out.append(_emit(idxs))
    assert len(out) == K and all(len(s) == seg_len for s in out), (
        f"expected {K} segments of len {seg_len}, got {[len(s) for s in out]}"
    )
    return out
