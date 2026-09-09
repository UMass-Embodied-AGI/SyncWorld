# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Gripperhead forward-dynamics SFT dataset.

Reads the "gripperhead" folder format shared by RoboCasa / LIBERO / RLBench
(``dataset_v3``) exports -- leaf ``expert/`` dirs that contain
``<cam>_rgb/video.mp4`` + ``pose.pkl`` (``gripper_matrix (T,4,4)`` +
``gripper_open (T,)``) (+ optional ``<cam>_mask/video.mp4`` and ``contacts.json``)
-- and emits Cosmos3 action samples for **forward-dynamics** training
(past video + future actions -> future video).

The data-loading + augmentation design is adapted from robotics-wan-dev
``training/rlbench_dataset_calib_segments_context.py`` (episode discovery,
imageio frame loading, axis augmentation, photometric augmentation, assert+skip
robustness, **null action for classifier-free guidance**, **sparse-history /
dense-prediction sampling**), but all calibration-segment logic is dropped. Each
clip is ``num_history_frames`` HISTORY frames sampled sparsely (every
``history_frame_stride`` real frames, ending at the anchor) + ``num_pred_frames``
PREDICTION frames sampled densely (stride 1) -- mirroring the original repo's
strided context + dense sample. The action is encoded
with Cosmos's own ``pose_abs_to_rel(..., rotation_format="rot6d")`` so it matches
the model's ee_pose action heads exactly: 10D ``[Δpos(3), Δrot6d(6), gripper(1)]``.

Segment sampling can be **contact-biased**: with ``contact_bias_prob`` the window
is shifted to contain a robot-object contact onset/release moment (read from
``contacts.json``; gripper keyframes are NOT used).

Null action (CFG): with ``p_null_action`` during training the whole action is
zeroed. An all-zero rot6d is an invalid/never-real rotation, so it is an
unambiguous "null"/unconditional signal -- the model learns it as the
action-CFG unconditional branch, and inference reproduces it for action guidance.

Folder-structure agnostic: discovery either reads a prebuilt
``episodes_cache.json`` (flat ``[{"path", "episode"}]``) or recursively walks
each root for ``expert`` leaves, so RoboCasa / LIBERO / RLBench all load through
the same class.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset

import imageio.v3 as iio

from cosmos_framework.data.vfm.action.action_spec import Gripper, Pos, Rot, build_action_spec
from cosmos_framework.data.vfm.action.calib_segments import build_calib_segment_indices
from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.pose_utils import compute_idle_frames, pose_abs_to_rel
from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline
from cosmos_framework.utils import log

# Single main-camera candidates (priority order); mirrors robotics-wan-dev
# `_resolve_rgb_video_path`. Wrist is a separate, optional view.
_RGB_CANDIDATES = (
    "agentview_rgb", "front_rgb", "base_camera_rgb", "render_camera_rgb", "left_rgb", "side_rgb", "rgb",
)
_WRIST_CANDIDATES = ("wrist_rgb", "eye_in_hand_rgb", "hand_rgb")

_MODE = "forward_dynamics"


# Task-neutral seed used whenever an episode has no REAL language instruction.
#
# It is deliberately NOT derived from the leaf path. The old behaviour turned the
# directory above ``episode_*`` into a pseudo-instruction (``PnPCounterToCab`` ->
# ``"pn p counter to cab"``), which is harmful in two ways:
#   * it teaches the text slot a SCENE/DOMAIN semantics ("you are in this folder")
#     while downstream the very same slot means a GOAL ("achieve this"), so the
#     mapping has to be un-learned by any downstream instruction-conditioned finetune;
#   * for sources whose directory above ``episode_*`` is a session id (e.g. DROID)
#     it is a per-episode identity leak.
# The seed is non-empty on purpose: ViewpointTextInfo / DurationFPSTextTimeStamps /
# ResolutionTextInfo all early-return on an empty caption (that is their CFG-null
# path), so an empty seed would silently drop the viewpoint / duration / FPS /
# resolution metadata too. See ``docs/wm2policy_external_literature_proposal.md``.
NEUTRAL_CAPTION_SEED = "A robot arm interacts with the scene."


def _caption_from_path(path: str) -> str:
    """Derive a caption from a dataset leaf path when no real instruction exists.

    LIBERO is the one source whose *filename* genuinely encodes the language
    instruction (``..._demo`` components), so that instruction is still
    recovered. Every other source falls back to :data:`NEUTRAL_CAPTION_SEED`
    rather than to its directory name.
    """
    parts = path.replace("\\", "/").split("/")
    # LIBERO: a REAL instruction is embedded in a "libero_..._demo" component.
    for part in parts:
        if "libero" in part.lower() and "__" in part:
            segment = part.split("__", 1)[1].removesuffix("_demo")
            tokens = segment.split("_")
            while tokens and (
                tokens[0].isupper()
                or tokens[0].lower().endswith("scene")
                or any(character.isdigit() for character in tokens[0])
            ):
                tokens.pop(0)
            if tokens:
                return " ".join(tokens).replace("_", " ").strip().lower()

    return NEUTRAL_CAPTION_SEED


def load_gripperhead_caption(leaf: str) -> str:
    """Load an episode's real language instruction, falling back to its path.

    New shards store ``lang`` in the leaf-local ``metadata.json``. Local RoboCasa
    renders made before that change keep the same value in the sibling
    ``base_init.pkl``, so accepting both formats lets them be repacked without a
    rerender.
    """

    metadata_path = os.path.join(leaf, "metadata.json")
    try:
        with open(metadata_path, encoding="utf-8") as file:
            metadata = json.load(file)
        if isinstance(metadata, dict):
            for key in ("lang", "instruction", "language_instruction", "ai_caption"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    for base_init_path in (
        os.path.join(leaf, "base_init.pkl"),
        os.path.join(os.path.dirname(leaf), "base_init.pkl"),
    ):
        try:
            with open(base_init_path, "rb") as file:
                base_init = pickle.load(file)
            value = base_init.get("lang") if isinstance(base_init, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, TypeError):
            pass

    return _caption_from_path(leaf)


def sample_history_real_frames(
    rng: random.Random,
    *,
    minimum: int,
    maximum: int,
    available: int,
) -> int:
    """Draw the number of real sparse-history frames used by training/eval."""
    minimum = max(0, min(int(minimum), int(maximum)))
    maximum = max(0, int(maximum))
    available = max(0, int(available))
    return min(rng.randint(minimum, maximum), available)


def build_padded_sparse_history_indices(
    *,
    anchor: int,
    num_history_frames: int,
    history_frame_stride: int,
    num_real_frames: int,
) -> list[int]:
    """Build training-style sparse history followed by last-frame tail padding."""
    history_length = max(0, int(num_history_frames))
    stride = max(1, int(history_frame_stride))
    available = max(0, int(anchor) // stride)
    num_real = max(0, min(int(num_real_frames), history_length, available))
    real = [int(anchor) - stride * (num_real - index) for index in range(num_real)]
    if not real:
        return [int(anchor)] * history_length
    return real + [real[-1]] * (history_length - len(real))


class GripperheadFDMDataset(Dataset):
    """Forward-dynamics samples from the gripperhead folder format.

    Each ``__getitem__`` returns a dict matching the Cosmos action contract
    (same shape as ``DROIDLeRobotDataset._build_result``); the surrounding
    :class:`ActionSFTDataset` + :class:`ActionTransformPipeline` then resize the
    video, build the forward-dynamics ``SequencePlan`` (with ``num_history_frames``
    conditioning frames), and pad the action to ``max_action_dim``.
    """

    # Whether the instruction-free (NEUTRAL_CAPTION_SEED) rule may apply to this dataset at all.
    # True here: this class IS the FDM/IDM pretraining source, which is instruction-free by design.
    # A downstream subclass that must keep real task language sets this to False -- see
    # RoboCasaNativePolicyDataset, whose "joint" is the 3-way fdm/idm/POLICY downstream mixture.
    _ALLOW_NEUTRAL_CAPTION: bool = True

    def __init__(
        self,
        roots: list[str] | str,
        *,
        # num_history_frames == 1 -> plain current+future (one joint VAE clip).
        # num_history_frames  > 1 -> history is a SEPARATE VAE item: H sparse (stride) frames
        #   strictly BEFORE the current frame (fully conditioning) + a current+future item
        #   ([current]+num_pred_frames) whose future latents are generated. H must be 4n+1 (e.g. 25).
        num_history_frames: int = 25,
        num_pred_frames: int = 16,
        history_frame_stride: int = 3,
        fps: float = 15.0,
        domain: str = "gripperhead",
        viewpoint: str = "third_person_view",
        use_wrist: bool = False,
        gripper_open_threshold: float = 0.6,
        is_train: bool = True,
        # --- task mode ---
        # "forward_dynamics" (default): past video + future actions -> generate future video (the
        #   pretraining recipe; actions are CLEAN conditioning, future video latents are the target).
        # "inverse_dynamics": observe ALL video frames -> predict the current+future action (video is
        #   CLEAN conditioning, the action tokens are the noised prediction target).
        # "joint": per-sample coin-flip between the two (prob ``joint_p_inverse`` of inverse) — each
        #   sample is forward OR inverse, never both in one sample. Mixed-mode batches are fine: the
        #   packer/noiser/loss act per-sample via each item's condition_mask (forward action tokens
        #   stay clean + unsupervised; inverse action tokens are noised + supervised).
        # All three are instruction-free pretraining modes -> every sample gets NEUTRAL_CAPTION_SEED
        # (see ``force_neutral_caption``).
        # The chosen mode is written into the sample's "mode" field; ActionTransformPipeline.
        # build_sequence_plan_from_mode turns it into the SequencePlan (which tokens condition vs
        # generate). Model, action heads and loss are shared across modes.
        task_mode: str = _MODE,
        joint_p_inverse: float = 0.5,  # task_mode="joint": P(a sample is inverse_dynamics); rest forward
        # --- action delta convention (abs gripper poses -> per-step action) ---
        # backward_framewise: T_i^-1 T_{i+1} (tiny per-frame deltas; the shipped recipe)
        # backward_anchored:  T_0^-1 T_{i+1} (everything relative to the current frame)
        action_pose_convention: str = "backward_framewise",
        # --- action representation scale: make each entry O(1) and centered via units/format.
        # euler_xyz + action_rotation_scale=57.2958 (rad->deg) + action_translation_scale=100
        # (m->cm) => 7D [Δpos_cm(3), Δeuler_deg(3), gripper(1)]. rot6d/1.0/1.0 gives the raw 10D
        # rot6d-delta form instead. ---
        action_rotation_format: str = "euler_xyz",
        action_translation_scale: float = 100.0,
        action_rotation_scale: float = 57.2958,
        # --- calibration segments. When on, each sample emits a LIST of vision+action items:
        # [calib_seg x K | history | current+future], each a SEPARATELY VAE-encoded conditioning
        # item with its own paired action block. Needs a sibling ``calibration/`` dir per episode. ---
        use_calibration: bool = True,
        calib_num_segments: int = 6,          # 6 (efficient, one per DoF) or 12 (both signs)
        calib_seg_len: int = 5,               # frames per calib segment (1+4n or 4n)
        calib_frame_interval: int = 3,        # subsample stride on the raw calib clip before detection
        calib_dirname: str = "calibration",
        p_include_history: float = 0.8,        # prob to include real history WHEN calib is present (calib-null forces history)
        p_include_calibration: float = 0.9,    # prob to include real calibration (else black null-placeholder + drop flag)
        # --- variable-length history/future sampling ---
        history_min_frames: int = 9,           # min REAL history frames when calib is null/dropped (0 when calib present)
        future_min_frames: int = 5,            # min REAL future frames to sample (rest last-frame-padded)
        # --- null action for classifier-free guidance (training only) ---
        p_null_action: float = 0.1,
        # --- contact-biased sampling (read from contacts.json) ---
        contact_bias_prob: float = 0.5,
        contact_filename: str = "contacts.json",
        # --- axis (pose-basis) augmentation; training only. One transform per sample, applied
        # consistently across calib / history / current+future poses. ---
        axis_aug_enabled: bool = True,
        axis_aug_mode: str = "all",  # all | swap_only | flip_only | scale_only | none
        p_pose_axis_aug: float = 0.6,
        axis_scale_min: float = 0.7,
        axis_scale_max: float = 1.5,
        # --- photometric augmentation (OFF by default; training only) ---
        photometric_aug_enabled: bool = False,
        # --- episode discovery ---
        episodes_cache_path: Optional[str] = None,
        include_counterfactual: bool = True,
        include_perturb: bool = False,
        dataset_multiplier: int = 1000,
        seed: int = 0,
        # --- history-video/action masking ---
        # With prob p_mask_history (training + multi-item path only), draw a recent contiguous run of the
        # TARGET modality's REAL history tokens and emit a per-sample override so the packer moves them out
        # of conditioning (they get noised + predicted): forward_dynamics => history VIDEO latents,
        # inverse_dynamics => history ACTION steps. Uses the known real-history count k_hist (no
        # latent-repeat detection). 0.0 => no override emitted.
        p_mask_history: float = 0.3,
    ) -> None:
        super().__init__()
        # Accept a list, or a single (comma-separated) string for env interpolation.
        if isinstance(roots, str):
            self.roots = [r.strip() for r in roots.split(",") if r.strip()]
        else:
            self.roots = list(roots)
        self.num_history_frames = int(num_history_frames)
        self.num_pred_frames = int(num_pred_frames)
        self.clip_len = self.num_history_frames + self.num_pred_frames
        # Cosmos VAE temporal compression is 4 -> clip length must be 4n+1.
        if (self.clip_len - 1) % 4 != 0:
            raise ValueError(
                f"num_history_frames+num_pred_frames must be 4n+1 (Cosmos VAE temporal compression), "
                f"got {self.clip_len} ({self.num_history_frames}+{self.num_pred_frames})."
            )
        # History is sampled SPARSELY (every `history_frame_stride` real frames) while the
        # prediction stays dense (stride 1): the clip is still `clip_len` sampled frames, but
        # the history covers a longer real-frame span. The real-frame span the window needs is
        # (H-1)*stride (sparse history, ending at the anchor) + num_pred_frames (dense future).
        self.history_frame_stride = max(1, int(history_frame_stride))
        # Variable-length sampling (robotics-wan parity): each sample draws k_hist / k_fut REAL frames
        # (>= these minimums) and last-frame-pads the rest, so each VAE item length stays FIXED (H, P+1).
        self.history_min_frames = max(0, min(int(history_min_frames), self.num_history_frames))
        self.future_min_frames = max(1, min(int(future_min_frames), self.num_pred_frames))
        if self.num_history_frames > 1:
            # History is a SEPARATE item: up to H sparse frames (stride S, entirely BEFORE current) +
            # a current+future item (1 current + up to P future). The ENCODED item length is fixed
            # (H=4n+1 -> e.g. 25->7 latents; P+1=4n+1 -> 17->5 latents); a short real sweep is
            # last-frame-padded to that length, so VAE encode_exact_durations are unchanged.
            if (self.num_history_frames - 1) % 4 != 0:
                raise ValueError(f"num_history_frames must be 1 or 4n+1 (e.g. 9), got {self.num_history_frames}")
            if self.num_pred_frames % 4 != 0:
                raise ValueError(f"num_history_frames>1 requires num_pred_frames=4n (so current+future=P+1 is 4n+1), got {self.num_pred_frames}")
            # An episode only needs the MINIMUM window now (min-history sparse frames + min-future);
            # anything shorter than the full sweep is last-frame-padded (clamped at sample time), so
            # short episodes are usable instead of being dropped for being < the full span.
            self.min_frames = self.history_min_frames * self.history_frame_stride + self.future_min_frames + 1
        else:
            # plain current+future: [current] + at least future_min_frames dense future.
            self.min_frames = self.future_min_frames + 1
        self.fps = float(fps)
        self.domain = domain
        self.domain_id = get_domain_id(domain)
        self.use_wrist = bool(use_wrist)
        # ``use_wrist`` stitches [main | wrist] side-by-side into one 2:1 frame, so the
        # emitted video is no longer a single third-person view. Report it as
        # ``concat_view`` so the prompt says "concatenated views from multiple camera
        # perspectives" instead of falsely claiming a plain third-person framing.
        if self.use_wrist and viewpoint != "concat_view":
            log.info(
                f"[GripperheadFDMDataset] use_wrist=True: overriding viewpoint {viewpoint!r} -> 'concat_view' "
                "(video is a [main | wrist] side-by-side stitch)"
            )
            viewpoint = "concat_view"
        self.viewpoint = viewpoint
        # Calibration segments (see class docstring / calib_segments.py).
        self.use_calibration = bool(use_calibration)
        self.calib_num_segments = int(calib_num_segments)
        self.calib_efficient = self.calib_num_segments == 6
        self.calib_seg_len = int(calib_seg_len)
        self.calib_frame_interval = max(1, int(calib_frame_interval))
        self.calib_dirname = str(calib_dirname)
        # float() FIRST: these can arrive as env-interpolated strings (e.g. "1.0") from the config,
        # and max(0.0, "1.0") would raise TypeError('>' not supported between str and float).
        self.p_include_history = min(1.0, max(0.0, float(p_include_history)))
        self.p_include_calibration = min(1.0, max(0.0, float(p_include_calibration)))
        if self.use_calibration and self.calib_num_segments not in (6, 12):
            raise ValueError(f"calib_num_segments must be 6 or 12, got {self.calib_num_segments}")
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.is_train = bool(is_train)
        self.action_pose_convention = str(action_pose_convention).strip()
        self.action_rotation_format = str(action_rotation_format).strip()
        self.action_translation_scale = float(action_translation_scale)
        self.action_rotation_scale = float(action_rotation_scale)
        if self.action_rotation_format not in ("rot6d", "euler_xyz", "quat_xyzw", "axisangle", "rot9d"):
            raise ValueError(
                f"action_rotation_format must be rot6d/euler_xyz/quat_xyzw/axisangle/rot9d, "
                f"got {self.action_rotation_format!r}"
            )
        if self.action_pose_convention not in ("backward_framewise", "backward_anchored"):
            raise ValueError(
                f"action_pose_convention must be one of backward_framewise/backward_anchored, "
                f"got {self.action_pose_convention!r}"
            )
        self.p_null_action = float(min(1.0, max(0.0, float(p_null_action))))

        self.contact_bias_prob = float(min(1.0, max(0.0, float(contact_bias_prob))))
        self.contact_filename = contact_filename
        self._contact_cache: dict[str, list[int]] = {}
        self._caption_cache: dict[str, str] = {}

        self.axis_aug_enabled = bool(axis_aug_enabled)
        self.axis_aug_mode = str(axis_aug_mode).lower().replace("-", "_")
        self.p_pose_axis_aug = float(min(1.0, max(0.0, float(p_pose_axis_aug))))
        self.axis_scale_min = float(axis_scale_min)
        self.axis_scale_max = float(axis_scale_max)

        self.photometric_aug_enabled = bool(photometric_aug_enabled)
        self._color_jitter = T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.03)

        # Task mode (forward_dynamics | inverse_dynamics | joint). "joint" flips a per-sample coin
        # between forward and inverse (prob joint_p_inverse); forward/inverse are fixed per run.
        # Only the emitted "mode" (-> SequencePlan) changes; model, heads and loss are shared.
        self.task_mode = str(task_mode).strip().lower()
        if self.task_mode not in ("forward_dynamics", "inverse_dynamics", "joint"):
            raise ValueError(
                f"task_mode must be 'forward_dynamics', 'inverse_dynamics' or 'joint', got {task_mode!r}"
            )
        # FDM/IDM pretraining is instruction-free BY DESIGN: the research setting introduces task
        # instructions only downstream. So EVERY sample emits NEUTRAL_CAPTION_SEED and a real
        # ``lang`` is never resolved -- not from metadata.json / base_init.pkl, not from a
        # LIBERO-style path. All three task modes here are pretraining modes, so this is
        # unconditional; ``_ALLOW_NEUTRAL_CAPTION`` remains the single opt-out hook for a
        # downstream subclass that must keep real task language.
        self.force_neutral_caption = self._ALLOW_NEUTRAL_CAPTION

        self.joint_p_inverse = float(min(1.0, max(0.0, float(joint_p_inverse))))
        # Modes whose ACTION is a prediction target (not clean conditioning): inverse (always) and
        # joint (on its per-sample inverse draws). Forward's action is always clean conditioning.
        _predicts_action = self.task_mode in ("inverse_dynamics", "joint")
        if self.task_mode == "inverse_dynamics" and self.p_null_action > 0:
            # Pure inverse: the ACTION is the prediction target, so the action-CFG null (zeroes the
            # pose delta + gripper=-1 sentinel) would corrupt the regression target — force it off.
            # (In joint, p_null_action is KEPT but applied ONLY to the per-sample forward draws; see
            # _load_sample.)
            log.info(
                f"[GripperheadFDMDataset] {self.task_mode}: forcing p_null_action=0 "
                "(action is the prediction target, not conditioning)"
            )
            self.p_null_action = 0.0
        if _predicts_action and self.axis_aug_enabled and self.p_pose_axis_aug > 0 and not self.use_calibration:
            # Pose axis-aug relabels the action's coordinate frame WITHOUT changing the input video, so
            # for inverse samples, without calibration (relabeled consistently AND observable), identical
            # observations map to randomly-different action targets -> an ill-posed target. Warn loudly.
            log.warning(
                "[GripperheadFDMDataset] inverse-dynamics samples + pose axis-aug WITHOUT calibration "
                "are ill-posed (axis-aug perturbs the TARGET action frame but not the observed video). "
                "Set gripperhead.p_pose_axis_aug=0 or enable gripperhead.use_calibration."
            )

        # Axis augmentation and wrist views are supported for the multi-item
        # path. Photometric augmentation is still wired only for the
        # single-item path, so fail loudly instead of silently no-op'ing.
        if (self.use_calibration or self.num_history_frames > 1) and self.photometric_aug_enabled:
            raise NotImplementedError(
                "photometric augmentation is not yet supported with the multi-item "
                "(history or calibration) construction; disable it or use num_history_frames=1 without calibration."
            )

        self.include_counterfactual = bool(include_counterfactual)
        self.include_perturb = bool(include_perturb)
        self.seed = int(seed)
        self.rng = random.Random(seed)
        self._seeded = False  # per-(rank, worker) RNG reseed happens lazily on first __getitem__
        self.dataset_multiplier = max(1, int(dataset_multiplier))

        self._to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])  # uint8 HWC -> CHW [0,1]
        self._spec = build_action_spec(Pos(), Rot(self.action_rotation_format), Gripper())

        self.episodes = self._discover_episodes(episodes_cache_path)
        if len(self.episodes) == 0:
            raise RuntimeError(f"No episodes discovered under roots={self.roots}")
        log.info(f"[GripperheadFDMDataset] {len(self.episodes)} episodes | clip={self.clip_len} "
                 f"(hist {self.num_history_frames}@stride{self.history_frame_stride} + pred {self.num_pred_frames}, "
                 f"span {self.min_frames}) | domain={self.domain}({self.domain_id}) "
                 f"| train={self.is_train} use_wrist={self.use_wrist} "
                 f"contact_bias={self.contact_bias_prob} p_null_action={self.p_null_action}")

        # Rank sharding. RankPartitionedDataLoader sets shard_rank / shard_world_size
        # (forwarded through the ActionSFTDataset wrapper); the DataLoader uses a
        # plain sequential sampler, so WITHOUT this every rank would read the same
        # episodes in the same order -> world_size x redundant reads.
        # _current_episodes() reshards lazily when they change.
        self.shard_rank = 0
        self.shard_world_size = 1
        self.shard_id = 0
        self._shard_key: Optional[tuple[int, int]] = None
        self._sharded_episodes = self.episodes

        # History-video/action masking probability. 0.0 => OFF: no RNG is consumed and no override is
        # emitted. See _load_sample_multiitem.
        self.p_mask_history = float(min(1.0, max(0.0, float(p_mask_history))))

    # ------------------------------------------------------------------ discovery
    def _resolve_rgb(self, leaf: str, candidates: tuple[str, ...]) -> Optional[str]:
        for name in candidates:
            p = os.path.join(leaf, name, "video.mp4")
            if os.path.isfile(p):
                return p
        return None

    def _is_valid_leaf(self, leaf: str) -> bool:
        return self._resolve_rgb(leaf, _RGB_CANDIDATES) is not None and os.path.isfile(os.path.join(leaf, "pose.pkl"))

    def _discover_episodes(self, cache_path: Optional[str]) -> list[dict[str, Any]]:
        # 1) explicit cache file
        if cache_path and os.path.isfile(cache_path):
            with open(cache_path) as f:
                raw = json.load(f)
            eps = [e for e in raw if self._keep_episode(e.get("episode", "expert"))]
            return [e for e in eps if self._is_valid_leaf(e["path"])]
        # 2) per-root cache, else recursive walk (and write a cache)
        episodes: list[dict[str, Any]] = []
        for root in self.roots:
            rc = os.path.join(root, "episodes_cache.json")
            if os.path.isfile(rc):
                with open(rc) as f:
                    raw = json.load(f)
                episodes += [e for e in raw if self._keep_episode(e.get("episode", "expert")) and self._is_valid_leaf(e["path"])]
                continue
            found: list[dict[str, Any]] = []
            for dirpath, _dirnames, _files in os.walk(root):
                base = os.path.basename(dirpath)
                if not self._keep_episode(base):
                    continue
                if self._is_valid_leaf(dirpath):
                    found.append({"path": dirpath, "episode": base})
            found.sort(key=lambda e: e["path"])
            try:
                with open(rc, "w") as f:
                    json.dump(found, f)
            except OSError:
                pass
            episodes += found
        episodes.sort(key=lambda e: e["path"])
        return episodes

    def _keep_episode(self, episode_name: str) -> bool:
        if episode_name == "expert":
            return True
        if self.include_counterfactual and episode_name.startswith("counterfactual"):
            return True
        # rlbench packs its non-expert augmentation rollouts as ``perturb_*`` (robosuite uses
        # ``counterfactual_*``); both are valid example clips (perturbed trajectory + matching
        # frames) per the video-based leaf rule. Opt in via ``include_perturb``.
        if self.include_perturb and episode_name.startswith("perturb"):
            return True
        return False

    # ------------------------------------------------------------------ pose / action
    def _load_pose(self, leaf: str) -> tuple[np.ndarray, np.ndarray]:
        """Load + validate gripper poses. Raises (-> sample skipped) on any
        malformed pose so a single bad episode never crashes the run."""
        pkl_path = os.path.join(leaf, "pose.pkl")
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"pose.pkl not found: {leaf}")
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        if "gripper_matrix" not in d:
            raise KeyError(f"pose.pkl missing 'gripper_matrix': {leaf}")
        mats = np.asarray(d["gripper_matrix"], dtype=np.float32)  # (T,4,4)
        gopen = np.asarray(d.get("gripper_open", np.ones(len(mats))), dtype=np.float32)  # (T,)
        if mats.ndim != 3 or mats.shape[1:] != (4, 4):
            raise ValueError(f"gripper_matrix has invalid shape {mats.shape}: {leaf}")
        if len(mats) != len(gopen):
            raise ValueError(f"gripper_matrix/gripper_open length mismatch ({len(mats)} vs {len(gopen)}): {leaf}")
        if not np.isfinite(mats).all():
            raise ValueError(f"non-finite gripper_matrix: {leaf}")
        gopen = (gopen > self.gripper_open_threshold).astype(np.float32)  # 1=open, 0=closed
        return mats, gopen

    def _sample_axis_transform(self) -> np.ndarray:
        """3x3 right-handed basis change S (det=+1) per `axis_aug_mode`."""
        mode = self.axis_aug_mode
        if mode in ("all", "both", "swap_and_flip", "swap_flip"):
            axes = [0, 1, 2]
            self.rng.shuffle(axes)
            signs = [(-1.0 if self.rng.random() < 0.5 else 1.0) for _ in range(2)]
            inv = sum(1 for i in range(3) for j in range(i + 1, 3) if axes[i] > axes[j])
            parity = -1.0 if inv % 2 else 1.0
            signs.append(parity / (signs[0] * signs[1]))
            S = np.zeros((3, 3), np.float32)
            for j, ax in enumerate(axes):
                S[ax, j] = signs[j]
            return S
        if mode in ("swap_only", "swap"):
            perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
            axes = perms[self.rng.randint(0, 2)]
            S = np.zeros((3, 3), np.float32)
            for j, ax in enumerate(axes):
                S[ax, j] = 1.0
            return S
        if mode in ("flip_only", "flip"):
            s1 = -1.0 if self.rng.random() < 0.5 else 1.0
            s2 = -1.0 if self.rng.random() < 0.5 else 1.0
            return np.diag(np.array([s1, s2, 1.0 / (s1 * s2)], np.float32))
        return np.eye(3, dtype=np.float32)

    def _sample_axis_st(self) -> Optional[tuple[np.ndarray, float]]:
        """Sample ONE axis-augmentation transform (body-frame relabel S, translation-scale s) for a
        sample, or None to skip. Drawn once per sample so every item (calibration / history /
        current+future) shares the SAME relabel — calibration only teaches the axis convention if the
        episode's actions are remapped the same way. Applied via _apply_axis_st as a RIGHT-multiply
        (R@S) so it SURVIVES the body-frame action delta (a LEFT/world-basis change would cancel to a
        no-op). Disabled at eval (``is_train=False``)."""
        if not self.is_train or not self.axis_aug_enabled or self.rng.random() >= self.p_pose_axis_aug:
            return None
        S = self._sample_axis_transform() if self.axis_aug_mode != "scale_only" else np.eye(3, np.float32)
        s = self.rng.uniform(self.axis_scale_min, self.axis_scale_max)
        return S, s

    @staticmethod
    def _apply_axis_st(poses_abs: np.ndarray, S: np.ndarray, s: float) -> np.ndarray:
        """Apply a BODY-frame axis relabel (RIGHT-multiply R@S, det S=+1) + scalar translation scale s
        to an abs-pose trajectory.

        RIGHT-multiply is deliberate (mirrors robotics-wan-dev ``einsum('tij,jk->tik', R, S)`` + ``t*scale``):
        the action is a body-frame relative delta ``T_i^-1 T_{i+1}``, which is INVARIANT to a world-frame
        (LEFT) basis change ``S@R`` -- ``(S R_i)^T (S R_{i+1}) = R_i^T R_{i+1}`` -- so left-multiplying makes
        axis-aug a silent NO-OP (only the scalar ``s`` survives). Right-multiplying instead conjugates the
        delta -- rotation ``-> S^T (R_i^T R_{i+1}) S``, translation ``-> s S^T (t_{i+1}-t_i)`` -- so the action
        axes are genuinely relabeled and the model must read the (identically-relabeled) calibration to decode
        the frame. Translation gets only the scalar ``s``; the per-step delta picks up ``S^T`` via the rotation."""
        out = poses_abs.copy()
        out[:, :3, :3] = poses_abs[:, :3, :3] @ S[None]      # R' = R @ S (body-frame relabel; survives the delta as S^T D S)
        out[:, :3, 3] = s * poses_abs[:, :3, 3]               # translation: scalar scale only (S enters the delta via R@S)
        return out

    def _maybe_axis_augment(self, poses_abs: np.ndarray) -> np.ndarray:
        """Single-clip path: sample + apply a per-call axis augmentation (or return unchanged)."""
        st = self._sample_axis_st()
        return poses_abs if st is None else self._apply_axis_st(poses_abs, *st)

    def _build_action(self, mats: np.ndarray, gopen: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """abs gripper poses -> per-step delta action [Δpos, Δrot, gripper].

        Scale is carried by ``action_rotation_format`` + ``action_translation_scale`` /
        ``action_rotation_scale`` (the shipped recipe: euler_xyz + cm + degrees => 7D), so no
        per-dim normalization table is involved."""
        mats = self._maybe_axis_augment(mats)
        poses_rel = pose_abs_to_rel(
            mats, rotation_format=self.action_rotation_format,
            pose_convention=self.action_pose_convention,
            translation_scale=self.action_translation_scale, rotation_scale=self.action_rotation_scale,
        )  # (L-1, 3+rot_dim)
        gripper = gopen[1:].reshape(-1, 1)  # (L-1, 1) gripper at the target frame of each framewise delta
        action = np.concatenate([poses_rel, gripper], axis=-1).astype(np.float32)  # (L-1, 3+rot_dim+1)
        initial_pose = torch.from_numpy(mats[0].copy()).float()
        return torch.from_numpy(action).float(), initial_pose

    # ------------------------------------------------------------------ frames
    @staticmethod
    def _read_video(video_path: str) -> np.ndarray:
        # Decode the whole (short) clip ONCE. Prefer the in-process **PyAV**
        # backend: imageio's default ffmpeg plugin spawns a subprocess per read
        # and intermittently fails with "Could not load meta information" under
        # sustained multi-worker concurrency. PyAV decodes in-process (no
        # subprocess) and is robust under load; fall back to the ffmpeg plugin
        # per-file if needed.
        try:
            return iio.imread(video_path, plugin="pyav")  # (T,H,W,C) uint8
        except Exception:
            return iio.imread(video_path)

    def _frames_to_tensor(self, frames_all: np.ndarray, indices: list[int]) -> torch.Tensor:
        n = len(frames_all)
        out = [self._to_tensor(Image.fromarray(frames_all[max(0, min(int(i), n - 1))])) for i in indices]
        return torch.stack(out, dim=0)  # (T,C,H,W) [0,1]

    def _center_square_crop(self, video: torch.Tensor) -> torch.Tensor:
        _, _, h, w = video.shape
        if h == w:
            return video
        m = min(h, w)
        top, left = (h - m) // 2, (w - m) // 2
        return video[:, :, top:top + m, left:left + m]

    def _maybe_photometric(self, video: torch.Tensor) -> torch.Tensor:
        if not self.is_train or not self.photometric_aug_enabled:
            return video
        video = self._color_jitter(video)
        video = (video + torch.randn_like(video) * 0.02).clamp(0.0, 1.0)
        return video

    # ------------------------------------------------------------------ sampling
    def _contact_moments(self, leaf: str, total: int) -> list[int]:
        """Frames where a robot-object contact begins (onset) or ends (release),
        derived from ``contacts.json`` (``per_frame_contacts`` -> any record with
        ``robot_object == True``). Gripper keyframes are NOT used. Cached per leaf."""
        if leaf in self._contact_cache:
            return [m for m in self._contact_cache[leaf] if 0 <= m < total]
        moments: list[int] = []
        cpath = os.path.join(leaf, self.contact_filename)
        if os.path.isfile(cpath):
            try:
                per_frame = json.load(open(cpath)).get("per_frame_contacts", []) or []
                has = [bool(any(r.get("robot_object") for r in (fr or []))) for fr in per_frame]
                # onset (False->True) and release (True->False) transitions
                moments = [i for i in range(1, len(has)) if has[i] != has[i - 1]]
            except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                moments = []
        self._contact_cache[leaf] = moments
        return [m for m in moments if 0 <= m < total]

    def _sample_window(self, leaf: str, total: int) -> list[int]:
        """Window indices: **sparse history** (every ``history_frame_stride`` real
        frames, ending at the anchor = the "current" frame) followed by a **dense
        prediction** (stride 1, from anchor+1). With ``contact_bias_prob`` the
        window is shifted so its real-frame span contains a contact onset/release
        moment. Returns ``clip_len`` strictly-increasing indices."""
        H, P, s = self.num_history_frames, self.num_pred_frames, self.history_frame_stride
        anchor_min = (H - 1) * s            # earliest anchor so the sparse history fits
        anchor_max = total - 1 - P          # latest anchor so the dense pred fits
        if anchor_max <= anchor_min:
            anchor = anchor_min             # episode just barely fits (caller enforces length)
        else:
            anchor = self.rng.randint(anchor_min, anchor_max)
            if self.contact_bias_prob > 0 and self.rng.random() < self.contact_bias_prob:
                moments = self._contact_moments(leaf, total)
                if moments:
                    m = self.rng.choice(moments)
                    # choose anchor so the moment lands inside [anchor-(H-1)*s, anchor+P]
                    lo = max(anchor_min, m - P)
                    hi = min(anchor_max, m + (H - 1) * s)
                    if lo <= hi:
                        anchor = self.rng.randint(lo, hi)
        history = [anchor - (H - 1 - i) * s for i in range(H)]  # sparse, ends at anchor
        pred = [anchor + 1 + j for j in range(P)]               # dense future
        return [max(0, min(int(x), total - 1)) for x in (history + pred)]

    # ------------------------------------------------------------------ caption
    @staticmethod
    def _caption_from_path(path: str) -> str:
        return _caption_from_path(path)

    def _caption_for_leaf(self, leaf: str) -> str:
        # FDM/IDM pretraining is instruction-free BY DESIGN (see ``force_neutral_caption``):
        # every episode gets the same task-neutral seed, even when the episode carries a real
        # ``lang``. Only a downstream subclass that clears ``_ALLOW_NEUTRAL_CAPTION`` reads it.
        if self.force_neutral_caption:
            return NEUTRAL_CAPTION_SEED
        caption = self._caption_cache.get(leaf)
        if caption is None:
            caption = load_gripperhead_caption(leaf)
            self._caption_cache[leaf] = caption
        return caption

    # ------------------------------------------------------------------ sharding
    def _current_episodes(self) -> list[dict[str, Any]]:
        """This rank's episode view. ``RankPartitionedDataLoader`` sets
        ``shard_rank`` / ``shard_world_size`` (forwarded from the
        ``ActionSFTDataset`` wrapper); each rank takes a disjoint stride of
        episodes so multi-GPU/-node training pulls ~1/world_size of the data
        instead of all of it. Recomputed only when the
        (rank, world_size) pair changes."""
        key = (int(self.shard_rank), int(self.shard_world_size))
        if self._shard_key != key:
            ws = max(1, key[1])
            r = key[0] % ws
            sub = self.episodes[r::ws] if ws > 1 else self.episodes
            # Never empty: if world_size > num_episodes some ranks would get [],
            # which would crash; fall back to the full pool for those ranks.
            self._sharded_episodes = sub if sub else self.episodes
            self._shard_key = key
            if ws > 1:
                log.info(
                    f"[GripperheadFDMDataset] shard rank {r}/{ws}: "
                    f"{len(self._sharded_episodes)}/{len(self.episodes)} episodes",
                    rank0_only=False,
                )
        return self._sharded_episodes

    def _maybe_reseed(self) -> None:
        """Reseed the window-sampling RNG once per (rank, worker) so different
        ranks / dataloader workers draw different windows + augmentations for the
        same episode. The dedicated ``random.Random(seed)`` is NOT touched by
        torch's per-worker seeding, so without this every worker would sample the
        identical window. Deterministic given (seed, shard_rank, worker_id)."""
        if self._seeded:
            return
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        self.rng = random.Random(self.seed + 1_000_003 * int(self.shard_rank) + 9_176 * int(wid))
        self._seeded = True

    # ------------------------------------------------------------------ dunder
    def __len__(self) -> int:
        # Equal across ranks (ceil(total / world_size)) so the per-rank DataLoaders
        # match length under DDP; the multiplier keeps the packer from hitting an
        # epoch boundary. __getitem__ wraps indices via ``% len(sharded_episodes)``.
        import math
        ws = max(1, int(getattr(self, "shard_world_size", 1)))
        per_rank = max(1, math.ceil(len(self.episodes) / ws))
        return per_rank * self.dataset_multiplier

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Resilient: skip bad / TOO-SHORT episodes instead of crashing the run.
        # Spread with a prime stride first (escape a local cluster of bad episodes
        # fast), then fall back to a dense linear sweep so EVERY episode is tried;
        # only a dataset with no loadable episode at all raises. (Previously it gave
        # up after 8 strided tries and took the whole job down when those 8 all
        # happened to be too-short counterfactual clips that can't satisfy the
        # history+prediction window.)
        self._maybe_reseed()
        eps = self._current_episodes()
        n = len(eps)
        start = int(index) % n
        last_err: Optional[Exception] = None
        warned = 0
        tried = set()
        spread = min(n, 64)
        for a in range(spread + n):  # a<spread: prime-stride spread; else: linear sweep of the rest
            idx = (start + a * 7919) % n if a < spread else (start + (a - spread)) % n
            if idx in tried:
                continue
            tried.add(idx)
            try:
                return self._load_sample(idx)
            except Exception as e:  # noqa: BLE001 - want to skip ANY bad sample
                last_err = e
                if warned < 16:  # a bad region can be large; cap the log spam
                    log.warning(
                        f"[GripperheadFDMDataset] skipping bad sample (episode "
                        f"{eps[idx]['path']}): {e!r}"
                    )
                    warned += 1
        raise RuntimeError(
            f"failed to load any valid sample after trying all {n} episodes "
            f"from index {index}: {last_err!r}"
        )

    def _choose_sample_mode(self) -> str:
        """Per-sample task mode. Fixed for forward_dynamics/inverse_dynamics; a per-sample coin-flip
        (prob ``joint_p_inverse`` of inverse) for "joint". Only "joint" draws from the RNG, so the
        fixed modes leave the RNG stream (window/aug/null draws) byte-identical to before."""
        if self.task_mode == "joint":
            return "inverse_dynamics" if self.rng.random() < self.joint_p_inverse else "forward_dynamics"
        return self.task_mode

    # ------------------------------------------------------------------ raw episode load
    def _raw_load(self, leaf: str) -> tuple:
        """Decode + validate ONE episode's raw arrays from disk:
        ``(mats, gopen, rgb_frames, total)``. Raises (-> the sample is skipped) on a bad,
        length-mismatched or TOO-SHORT episode."""
        mats, gopen = self._load_pose(leaf)  # validated (raises -> skip)

        rgb_path = self._resolve_rgb(leaf, _RGB_CANDIDATES)
        if rgb_path is None:
            raise FileNotFoundError(f"no RGB video: {leaf}")
        rgb_frames = self._read_video(rgb_path)  # (Tv,H,W,C) uint8 — one decode

        # Robustness (robotics-wan-dev style): require video and pose to agree in
        # length, and enough frames for one contiguous window; else skip.
        n_pose, n_rgb = len(mats), len(rgb_frames)
        if n_pose != n_rgb:
            raise ValueError(f"pose/RGB length mismatch ({n_pose} vs {n_rgb}): {leaf}")
        if n_rgb < self.min_frames:
            raise ValueError(
                f"episode too short ({n_rgb} < required span {self.min_frames} for "
                f"hist {self.num_history_frames}@stride{self.history_frame_stride} + pred {self.num_pred_frames}): {leaf}"
            )
        total = n_rgb
        return mats, gopen, rgb_frames, total

    def _load_sample(self, index: int) -> dict[str, Any]:
        eps = self._current_episodes()
        epi = eps[int(index) % len(eps)]
        leaf = epi["path"]
        mats, gopen, rgb_frames, total = self._raw_load(leaf)

        # Per-sample task mode (fixed, or a joint coin-flip). Drives the emitted "mode" and gates the
        # action-CFG null below (null only applies to forward samples, where the action is conditioning).
        mode = self._choose_sample_mode()

        # Multi-item construction ([calib x K?] + [history?] + [current+future]) whenever calibration
        # is on or history is a separate item (num_history_frames > 1). Otherwise fall through to the
        # plain current+future joint clip (single VAE item) below.
        if self.use_calibration or self.num_history_frames > 1:
            return self._load_sample_multiitem(leaf, epi, mats, gopen, rgb_frames, total, mode)

        indices = self._sample_window(leaf, total)

        # RGB view (main + optional wrist below), square. Photometric aug on RGB only.
        video = self._build_view(leaf, rgb_frames, _WRIST_CANDIDATES, indices)
        video = self._maybe_photometric(video)

        action, initial_pose = self._build_action(mats[indices], gopen[indices])  # (clip_len-1, act_dim)

        idle_frames = compute_idle_frames(
            action, self._spec,
            eps_t=5e-3 / self.fps, eps_r=np.deg2rad(1.5) / self.fps, eps_g=1e-2,
            joint_threshold=5e-3 / self.fps, min_streak=3,
        )

        # Null action (classifier-free guidance): during training, with p_null_action, emit the null
        # sentinel (6D delta zeroed + gripper=-1). See _null_action: with 7D euler an all-zero action
        # is a valid stay-still action (ambiguous), so the gripper=-1 sentinel marks the unconditional
        # branch unambiguously.
        if self.is_train and self.p_null_action > 0 and mode == "forward_dynamics" and self.rng.random() < self.p_null_action:
            action = self._null_action(action)

        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)  # (C,T,H,W) uint8
        return {
            "ai_caption": self._caption_for_leaf(leaf),
            "video": formatted_video,
            "action": action,
            "conditioning_fps": torch.tensor(self.fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "viewpoint": self.viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            "initial_pose": initial_pose,
            "path": leaf,
            "episode": epi.get("episode", "expert"),
        }

    def _build_view(self, leaf: str, main_frames: np.ndarray, wrist_cands: tuple[str, ...],
                    indices: list[int]) -> torch.Tensor:
        """One square view, or ``[main | wrist]`` side-by-side (2:1 wide) when ``use_wrist``.

        Each view is center-square-cropped BEFORE the concat (cropping the concatenated pair would
        slice through both). A missing wrist video falls back to a BLACK right half so the item shape
        stays consistent with the wide tier (shards can't be pre-filtered on wrist presence)."""
        vid = self._center_square_crop(self._frames_to_tensor(main_frames, indices))  # (L,C,H,H) [0,1]
        if self.use_wrist:
            wp = self._resolve_rgb(leaf, wrist_cands)
            wrist = (self._center_square_crop(self._frames_to_tensor(self._read_video(wp), indices))
                     if wp is not None else torch.zeros_like(vid))
            vid = self._concat_views(vid, wrist)
        return vid

    @staticmethod
    def _concat_views(main: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        """``[main | wrist]`` SIDE-BY-SIDE -> (L,C,H,2W): main LEFT, wrist RIGHT.

        Pairs with the wide ``"<res>_w2"`` tier, so each view keeps a full undistorted square."""
        import torch.nn.functional as F
        _, _, h, w = main.shape
        if wrist.shape[-2:] != (h, w):
            wrist = F.interpolate(wrist, size=(h, w), mode="bilinear", align_corners=False)
        return torch.cat([main, wrist], dim=-1)

    # ------------------------------------------------------------------ calibration (multi-item)
    def _frames_to_item(
        self,
        main_frames: np.ndarray,
        indices: list[int],
        *,
        leaf: Optional[str] = None,
    ) -> torch.Tensor:
        """Build one multi-item vision clip, preserving the configured view layout."""
        if leaf is None:
            vid = self._center_square_crop(self._frames_to_tensor(main_frames, indices))
        else:
            # History/current-future items must use the same main+wrist layout as the single-item path.
            vid = self._build_view(leaf, main_frames, _WRIST_CANDIDATES, indices)
        return (vid * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)  # (C,L,H,W)

    def _action_from_poses(self, mats_slice: np.ndarray, gopen_slice: np.ndarray) -> torch.Tensor:
        """Per-frame delta action for one item (``len`` poses -> ``len-1`` deltas)."""
        poses_rel = pose_abs_to_rel(
            mats_slice, rotation_format=self.action_rotation_format,
            pose_convention=self.action_pose_convention,
            translation_scale=self.action_translation_scale, rotation_scale=self.action_rotation_scale,
        )  # (L-1, 3+rot_dim)
        grip = gopen_slice[1:].reshape(-1, 1)
        return torch.from_numpy(np.concatenate([poses_rel, grip], axis=-1).astype(np.float32))  # (L-1, 3+rot_dim+1)

    @staticmethod
    def _pad_last(indices: list[int], target_len: int, fallback_idx: int) -> list[int]:
        """Resize a frame-index list to ``target_len`` by repeating the LAST index at the back
        (robotics-wan ``_pad_with_last``). Empty -> all ``fallback_idx``; longer -> truncate."""
        if target_len <= 0:
            return []
        if len(indices) == 0:
            return [int(fallback_idx)] * target_len
        if len(indices) >= target_len:
            return list(indices[:target_len])
        return list(indices) + [indices[-1]] * (target_len - len(indices))

    def _null_action(self, action: torch.Tensor) -> torch.Tensor:
        """Null action for action-CFG. The 6D pose delta is zeroed but the gripper (last dim) is set
        to a -1 SENTINEL: with 7D euler an all-zero action is a VALID "stay-still" (identity) action
        that collides with real idle frames, so zeroing alone is an ambiguous null. gripper is
        binarized to {0,1}, so -1 is an unambiguous out-of-distribution null marker (robotics-wan uses
        a dedicated flag dim for the same purpose). Preserved across rot6d/euler (last dim = gripper)."""
        out = torch.zeros_like(action)
        if out.shape[-1] > 0:
            out[..., -1] = -1.0
        return out

    def _resolve_calib_dir(self, leaf: str) -> Optional[str]:
        """Find the calibration dir: ``<leaf>/calibration`` (eval layout) or the parent's (shard unit)."""
        for c in (os.path.join(leaf, self.calib_dirname),
                  os.path.join(os.path.dirname(leaf.rstrip("/")), self.calib_dirname)):
            if (os.path.isdir(c) and self._resolve_rgb(c, _RGB_CANDIDATES) is not None
                    and os.path.isfile(os.path.join(c, "pose.pkl"))):
                return c
        return None

    def _build_calib_items(self, calib_dir: str, axis_st: Optional[tuple[np.ndarray, float]] = None,
                           ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Load + segment the calibration clip -> (K video items, K action items).

        Each calib segment is one conditioning vision item + one paired (fully-conditioning)
        action block. Segmentation via calib_segments.build_calib_segment_indices (monotone-axis
        runs; ``move_range['movement_order']`` fixes per-axis sweep sign). ``axis_st`` (the sample's
        shared axis-augmentation transform) is applied to the calib POSES for the action blocks so
        they match the episode's augmented actions; segmentation still runs on the ORIGINAL poses so
        the per-axis run detection + move_order signs stay valid, and the calib VIDEO is untouched."""
        calib_rgb = self._read_video(self._resolve_rgb(calib_dir, _RGB_CANDIDATES))
        d = pickle.load(open(os.path.join(calib_dir, "pose.pkl"), "rb"))
        cmats = np.asarray(d["gripper_matrix"], dtype=np.float32)
        copen = (np.asarray(d.get("gripper_open", np.ones(len(cmats))), dtype=np.float32)
                 > self.gripper_open_threshold).astype(np.float32)
        move_order = None
        mrp = os.path.join(calib_dir, "move_range.pkl")
        if os.path.isfile(mrp):
            try:
                move_order = pickle.load(open(mrp, "rb")).get("movement_order")
            except Exception:
                move_order = None
        n = min(len(cmats), len(calib_rgb))
        sub = list(range(0, n, self.calib_frame_interval)) or [0]
        seg_lists = build_calib_segment_indices(cmats[sub], self.calib_seg_len, self.calib_efficient, move_order)
        cmats_act = self._apply_axis_st(cmats, *axis_st) if axis_st is not None else cmats  # augmented poses for actions
        vids, acts = [], []
        for seg in seg_lists:
            real = [sub[i] for i in seg]  # map subsampled indices back to raw calib frame indices
            vids.append(self._frames_to_item(calib_rgb, real, leaf=calib_dir))
            acts.append(self._action_from_poses(cmats_act[real], copen[real]))
        return vids, acts

    # ------------------------------------------------------------------ multi-item window sampler
    def _sample_anchor(self, leaf: str, total: int, mode: str) -> int:
        """Pick the current-frame anchor for the multi-item window. Placed so at least
        history_min_frames sparse history frames fit BEFORE it. Evaluation samples require the
        complete future horizon; training may use future_min_frames."""
        S, P = self.history_frame_stride, self.num_pred_frames
        required_future = P if not self.is_train else self.future_min_frames
        if self.is_train:
            anchor_min = self.history_min_frames * S
            anchor_max = total - 1 - required_future
        else:
            anchor_min = self.num_history_frames * S
            anchor_max = total - 1 - required_future
        if anchor_max <= anchor_min:
            return max(0, min(anchor_min, total - 1))
        anchor = self.rng.randint(anchor_min, anchor_max)
        if self.contact_bias_prob > 0 and self.rng.random() < self.contact_bias_prob:
            moments = self._contact_moments(leaf, total)
            if moments:
                m = self.rng.choice(moments)
                lo = max(anchor_min, m - P)   # keep the contact within the (dense) future span
                hi = min(anchor_max, m)
                if lo <= hi:
                    anchor = self.rng.randint(lo, hi)
        return anchor

    def _sample_future_length(self, *, mode: str, available: int) -> int:
        """Return the number of real future frames; evaluation always gets the full horizon."""
        requested = (
            self.num_pred_frames
            if not self.is_train
            else self.rng.randint(self.future_min_frames, self.num_pred_frames)
        )
        return max(1, min(requested, available))

    def _pick_history_mask(self, *, k_hist: int, mode: str) -> Optional[dict]:
        """Feature A (history-video/action masking). With prob ``p_mask_history``, draw a RECENT
        contiguous run of the TARGET modality's REAL history tokens to move out of conditioning (they
        get noised + predicted by the packer). Returns a per-sample override dict, or ``None`` when the
        feature is off / not applicable (=> byte-identical, no RNG consumed).

        Target modality by mode: ``forward_dynamics`` masks history VIDEO latents (the model must infer
        history video from the action stream, not copy it); ``inverse_dynamics`` masks history ACTION
        steps (the model must read the video to infer action, not extrapolate the history action).

        The override is ``{"item_index", "vision_frames", "action_steps"}`` where ``item_index`` is the
        LOCAL index of the history item within the per-sample item list (= number of prepended calib
        items). Uses the known real-history count ``k_hist`` — no latent-repeat detection.
        """
        if not (self.is_train and self.p_mask_history > 0.0 and self.num_history_frames > 1):
            return None
        if int(k_hist) <= 0:
            return None
        if mode not in ("forward_dynamics", "inverse_dynamics"):
            return None
        if self.rng.random() >= self.p_mask_history:
            return None
        hist_item_index = self.calib_num_segments if self.use_calibration else 0
        if mode == "inverse_dynamics":
            # Target = ACTION. History action item has (H-1) steps; the REAL (non-padding) deltas are the
            # transitions between the k_hist real frames => (k_hist - 1). Mask a recent contiguous run.
            n_real = max(0, int(k_hist) - 1)
            if n_real < 1:
                return None
            seg_len = self.rng.randint(1, n_real)
            seg_end = n_real - 1  # most-recent real action step
            seg_start = seg_end - seg_len + 1
            return {"item_index": int(hist_item_index), "vision_frames": [],
                    "action_steps": list(range(seg_start, seg_end + 1))}
        # forward_dynamics: Target = VIDEO. History video item VAE-encodes H frames -> (H-1)//4+1 latents;
        # fully-real latents (all constituent frames real) = 1 + (k_hist - 1)//4. Mask a recent contiguous run.
        n_lat = (self.num_history_frames - 1) // 4 + 1
        n_real = min(n_lat, 1 + (int(k_hist) - 1) // 4)
        if n_real < 1:
            return None
        seg_len = self.rng.randint(1, n_real)
        seg_end = n_real - 1  # most-recent real history latent
        seg_start = seg_end - seg_len + 1
        return {"item_index": int(hist_item_index),
                "vision_frames": list(range(seg_start, seg_end + 1)), "action_steps": []}

    def _load_sample_multiitem(self, leaf, epi, mats, gopen, rgb_frames, total, mode) -> dict[str, Any]:
        """Multi-item sample ``[calib x K?] + [history?] + [current+future]`` — each entry is a
        SEPARATE VAE-encoded item with its own fully-conditioning action block, EXCEPT the final
        current+future item, whose future latents are the only ones generated. History (H sparse
        frames strictly BEFORE the current frame) is included when num_history_frames > 1;
        calibration segments are prepended when use_calibration. One window sampler +
        _frames_to_item / _action_from_poses build every item."""
        anchor = self._sample_anchor(leaf, total, mode)
        H, P, S = self.num_history_frames, self.num_pred_frames, self.history_frame_stride
        clamp = lambda x: max(0, min(int(x), total - 1))

        # Axis augmentation: sample ONE transform per sample and apply it to the episode poses here
        # AND to the calibration poses below (in _build_calib_items), so calibration / history /
        # current+future all share the same world-basis change (a no-op when axis aug is off / eval).
        axis_st = self._sample_axis_st()
        if axis_st is not None:
            mats = self._apply_axis_st(mats, *axis_st)

        # Calibration inclusion decided FIRST — it drives the history coupling below (a null/dropped
        # calibration FORCES history, so the sample still carries embodiment motion cues).
        calib_present = False
        calib_v = calib_a = None
        if self.use_calibration:
            calib_dir = self._resolve_calib_dir(leaf)
            include_calib = calib_dir is not None and ((not self.is_train) or (self.rng.random() < self.p_include_calibration))
            if include_calib:
                try:
                    calib_v, calib_a = self._build_calib_items(calib_dir, axis_st)
                    calib_present = True
                except Exception as e:  # noqa: BLE001 - fall back to null placeholder on any calib read error
                    log.warning(f"[GripperheadFDMDataset] calib load failed ({calib_dir}): {e!r}; using null placeholder")
                    calib_v = calib_a = None

        # Policy has no future-length input, so it always uses all P real frames. Random future
        # truncation remains available only to forward/inverse training, where it is observable or
        # otherwise part of the existing recipe.
        available_future = total - 1 - anchor
        try:
            k_fut = self._sample_future_length(mode=mode, available=available_future)
        except ValueError as error:
            raise ValueError(f"{error}: {leaf}") from error
        fut_real = [anchor + 1 + j for j in range(k_fut)]
        cf_idx = [clamp(anchor)] + [clamp(x) for x in self._pad_last(fut_real, P, anchor + k_fut)]
        cf_vid = self._frames_to_item(
            rgb_frames, cf_idx, leaf=leaf
        )  # (C,P+1,h,w) -> P//4+1 latents
        cf_act = self._action_from_poses(mats[cf_idx], gopen[cf_idx])        # (P, D) — future-driving action
        if self.is_train and self.p_null_action > 0 and mode == "forward_dynamics" and self.rng.random() < self.p_null_action:
            cf_act = self._null_action(cf_act)  # action-CFG null (sentinel) on the future-driving action only
        C_, _, h_, w_ = cf_vid.shape

        video_items: list[torch.Tensor] = []
        action_items: list[torch.Tensor] = []
        history_mask: Optional[dict] = None  # Feature A override (None => no masking, byte-identical)

        # calibration items: real segments, or black null placeholders (calibration-CFG).
        if self.use_calibration:
            if calib_v is None:
                calib_v = [torch.zeros((C_, self.calib_seg_len, h_, w_), dtype=torch.uint8)
                           for _ in range(self.calib_num_segments)]
                calib_a = [torch.zeros((max(1, self.calib_seg_len) - 1, cf_act.shape[1]), dtype=torch.float32)
                           for _ in range(self.calib_num_segments)]
            video_items += calib_v
            action_items += calib_a

        # history item (only when num_history_frames > 1). Coupling (robotics-wan parity):
        #   - calibration REAL this sample  -> history OPTIONAL (prob p_include_history), min 0 frames;
        #   - calibration NULL/dropped/off  -> history FORCED, >= history_min_frames real frames.
        # Draw k_hist MOST-RECENT sparse frames (stride S, before current), last-frame-pad the tail to H
        # (all = current frame when k_hist == 0). The ENCODED item length stays H.
        if self.num_history_frames > 1:
            if calib_present:
                include_hist = (not self.is_train) or (self.rng.random() < self.p_include_history)
                hist_min = 0
            else:
                include_hist = True
                hist_min = self.history_min_frames
            if not include_hist:
                k_hist = 0
            elif not self.is_train:
                k_hist = min(H, anchor // S)
            else:
                k_hist = sample_history_real_frames(
                    self.rng,
                    minimum=hist_min,
                    maximum=H,
                    available=anchor // S,
                )
            hist_idx = build_padded_sparse_history_indices(
                anchor=anchor,
                num_history_frames=H,
                history_frame_stride=S,
                num_real_frames=k_hist,
            )
            hist_vid = self._frames_to_item(rgb_frames, hist_idx, leaf=leaf)
            hist_act = self._action_from_poses(mats[hist_idx], gopen[hist_idx])
            video_items.append(hist_vid)
            action_items.append(hist_act)
            # Feature A: pick a recent contiguous run of the target modality's REAL history tokens to
            # move out of conditioning (predicted). None unless ``p_mask_history`` > 0 fires.
            history_mask = self._pick_history_mask(k_hist=k_hist, mode=mode)

        video_items.append(cf_vid)
        action_items.append(cf_act)

        idle_frames = compute_idle_frames(
            cf_act, self._spec,
            eps_t=5e-3 / self.fps, eps_r=np.deg2rad(1.5) / self.fps, eps_g=1e-2,
            joint_threshold=5e-3 / self.fps, min_streak=3,
        )
        return {
            "ai_caption": self._caption_for_leaf(leaf),
            "video": video_items,
            "action": action_items,
            "conditioning_fps": torch.tensor(self.fps, dtype=torch.long),
            "mode": mode,
            # Feature A override (None => no history masking). Consumed + popped in
            # ActionTransformPipeline._call_multiitem so it never reaches collation.
            "history_mask": history_mask,
            # Number of leading CALIBRATION items in this sample's item list (0 when calib is off). Used
            # by the model's paired-distillation (Feature B) to null the student's calibration; harmless
            # otherwise. Always the first `num_calib_items` items are calibration.
            "num_calib_items": torch.tensor(
                self.calib_num_segments if self.use_calibration else 0, dtype=torch.long
            ),
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "viewpoint": self.viewpoint,
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            "initial_pose": torch.from_numpy(mats[anchor].copy()).float(),
            "path": leaf,
            "episode": epi.get("episode", "expert"),
        }


def get_gripperhead_fdm_sft_dataset(
    *,
    roots: list[str] | str = "",
    resolution: str | int = 512,
    num_history_frames: int = 25,
    num_pred_frames: int = 16,
    history_frame_stride: int = 3,
    fps: float = 15.0,
    domain: str = "gripperhead",
    use_wrist: bool = False,
    is_train: bool = True,
    # Task mode: "forward_dynamics" | "inverse_dynamics" | "joint" (per-sample coin-flip).
    # See GripperheadFDMDataset. joint_p_inverse = P(a sample is inverse) when task_mode="joint".
    task_mode: str = "forward_dynamics",
    joint_p_inverse: float = 0.5,
    # --- action delta convention (backward_framewise | backward_anchored) ---
    action_pose_convention: str = "backward_framewise",
    # --- action representation scale: euler_xyz + cm + degrees => 7D (the shipped recipe) ---
    action_rotation_format: str = "euler_xyz",
    action_translation_scale: float = 100.0,
    action_rotation_scale: float = 57.2958,
    # --- calibration segments ---
    use_calibration: bool = True,
    calib_num_segments: int = 6,
    calib_seg_len: int = 5,
    calib_frame_interval: int = 3,
    p_include_history: float = 0.8,
    p_include_calibration: float = 0.9,
    history_min_frames: int = 9,
    future_min_frames: int = 5,
    p_null_action: float = 0.1,
    contact_bias_prob: float = 0.5,
    axis_aug_enabled: bool = True,
    axis_aug_mode: str = "all",
    p_pose_axis_aug: float = 0.6,
    photometric_aug_enabled: bool = False,
    include_counterfactual: bool = True,
    include_perturb: bool = False,
    dataset_multiplier: int = 1000,
    episodes_cache_path: Optional[str] = None,
    max_action_dim: int = 64,
    cfg_dropout_rate: float = 0.1,
    tokenizer_config: dict | None = None,
    append_viewpoint_info: bool = True,
    append_duration_fps_timestamps: bool = True,
    append_resolution_info: bool = True,
    seed: int = 0,
    # History-video/action masking. With prob p_mask_history a recent contiguous run of the TARGET
    # modality's REAL history tokens is moved out of conditioning (predicted): forward => history
    # VIDEO latents, inverse => history ACTION steps. Only active on the multi-item path
    # (num_history_frames > 1) during training; 0.0 = OFF.
    p_mask_history: float = 0.3,
) -> ActionSFTDataset:
    """Build the gripperhead forward/inverse-dynamics SFT dataset (RoboCasa / LIBERO / RLBench).

    Defaults mirror the shipped pretraining recipe, whose single source of truth is
    ``GripperheadConfig`` in ``cosmos_framework/configs/toml_config/sft_config.py`` (the
    ``[gripperhead]`` TOML block). ``gripperhead_recipe_defaults_test.py`` pins the two together.

    ``num_history_frames`` past frames condition the generation of ``num_pred_frames`` future
    frames. The history is sampled SPARSELY (every ``history_frame_stride`` real frames) and the
    prediction DENSELY (stride 1), so the conditioning covers a longer past at a lower temporal rate.

    Notable knobs:
    - ``contact_bias_prob``: probability the window is shifted to contain a robot-object contact
      onset/release moment (from ``contacts.json``).
    - ``p_null_action``: probability (training only) the action is zeroed -> the model's action-CFG
      unconditional branch.
    - ``use_wrist``: stitch ``[main | wrist]`` side-by-side (2:1) and switch to the wide
      ``"<res>_w2"`` tier.
    - ``is_train=False`` disables null-action, axis aug and photometric aug (use for evaluation).
    """
    def _as_bool(x: Any) -> bool:
        return x if isinstance(x, bool) else str(x).strip().lower() in ("1", "true", "yes", "y", "on")
    use_wrist = _as_bool(use_wrist)
    is_train = _as_bool(is_train)
    use_calibration = _as_bool(use_calibration)
    calib_num_segments = int(calib_num_segments)

    # A side-by-side [main | wrist] frame is 2:1 wide -> use the dedicated wide tier so the transform
    # resizes to <res>h x 2*<res>w (e.g. 256 -> 256x512) instead of square.
    if use_wrist:
        wide = f"{resolution}_w2"
        from cosmos_framework.data.vfm.utils import VIDEO_RES_SIZE_INFO
        if wide not in VIDEO_RES_SIZE_INFO:
            raise ValueError(
                f"use_wrist=True needs a wide resolution tier '{wide}' in VIDEO_RES_SIZE_INFO; "
                f"available: {[k for k in VIDEO_RES_SIZE_INFO if k.endswith('_w2')]}"
            )
        resolution = wide

    if not roots:
        raise ValueError("get_gripperhead_fdm_sft_dataset: `roots` is required (comma-separated is OK).")
    dataset = GripperheadFDMDataset(
        roots=roots,
        episodes_cache_path=episodes_cache_path,
        num_history_frames=num_history_frames,
        num_pred_frames=num_pred_frames,
        history_frame_stride=history_frame_stride,
        fps=fps,
        domain=domain,
        use_wrist=use_wrist,
        is_train=is_train,
        task_mode=str(task_mode).strip().lower(),
        joint_p_inverse=float(joint_p_inverse),
        action_pose_convention=str(action_pose_convention).strip(),
        action_rotation_format=str(action_rotation_format).strip(),
        action_translation_scale=float(action_translation_scale),
        action_rotation_scale=float(action_rotation_scale),
        use_calibration=use_calibration,
        calib_num_segments=calib_num_segments,
        calib_seg_len=calib_seg_len,
        calib_frame_interval=calib_frame_interval,
        p_include_history=p_include_history,
        p_include_calibration=p_include_calibration,
        history_min_frames=history_min_frames,
        future_min_frames=future_min_frames,
        p_null_action=p_null_action,
        contact_bias_prob=contact_bias_prob,
        axis_aug_enabled=axis_aug_enabled,
        axis_aug_mode=axis_aug_mode,
        p_pose_axis_aug=p_pose_axis_aug,
        photometric_aug_enabled=photometric_aug_enabled,
        include_counterfactual=include_counterfactual,
        include_perturb=include_perturb,
        dataset_multiplier=dataset_multiplier,
        seed=seed,
        p_mask_history=float(p_mask_history),
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        num_history_frames=num_history_frames,
        append_viewpoint_info=append_viewpoint_info,
        append_duration_fps_timestamps=append_duration_fps_timestamps,
        append_resolution_info=append_resolution_info,
    )
    return ActionSFTDataset(dataset, transform, resolution)
