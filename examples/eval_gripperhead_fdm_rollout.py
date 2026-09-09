#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Regular full-episode evaluation for the gripperhead forward-dynamics recipe.

Every eval episode is rolled out end-to-end, ``--num-pred-frames`` (16) at a time. The conditioning
layout mirrors the training dataset and is selected purely by the flags:

  - ``--num-history-frames 1``: each segment is ONE joint clip [current] + 16 future frames; condition
    on the current-frame latent and generate the future.
  - ``--num-history-frames N`` (N = 4n+1, e.g. 25): history is a SEPARATE VAE item — N sparse frames,
    every ``--history-frame-stride`` frames, strictly BEFORE the current frame — plus a current+future
    item; the model conditions on both and generates the future.
  - ``--use-calibration`` additionally PREPENDS the K per-DoF calibration segments
    (``--calib-seg-len`` frames each) as fully-conditioning items, giving
    ``[calib x K] (+ [history]) + [current+future]`` — the same multi-item sample the training dataset
    builds. The calib clip is episode-fixed, so its K items are built ONCE per episode and reused
    across all of that episode's segments.

ROLLOUT (``--num-rollout-rounds N``): per segment start ``s`` (every stride = num_pred_frames), run N
autoregressive rounds — round 0 is teacher-forced (history = GT), each later round feeds the model's
OWN generated frames back as history (closed-loop) — then stitch the rounds (dropping the 1-frame
overlap) into one (pred, gt) pair. N=1 is pure teacher forcing; N=2 (the default) surfaces one step of
error accumulation. Actions are always the GT per-frame delta poses, built to match training.

Every flag documented as "MUST match the trained checkpoint" has to agree with the ``[gripperhead]``
block the checkpoint was trained under; those defaults ARE the shipped training recipe, so a
checkpoint from the default training config needs none of them.

The SAMPLING defaults (``--num-steps 20``, ``--action-cfg-scale 1.0``, ``--compile``) are instead a
measured fast preset: ~3x faster than the reference sampler settings for -0.19 dB PSNR, with SSIM
and LPIPS marginally better. To reproduce the reference numbers exactly, pass
``--num-steps 35 --action-cfg-scale 5.0 --no-compile``.

OUTPUT: ``<out>/<tag>/<view>/episode_{idx:04d}/seg{seg:04d}_start{frame:06d}.mp4`` ([GT|pred]
side-by-side) + ``<out>/<tag>/<view>/metrics_full_episode.csv`` (per-segment rows + a final 'average'
row; columns episode,segment_index,start_frame,num_frames_eval,psnr,ssim,lpips — PSNR/SSIM are ported
verbatim from robotics-wan so the values are directly comparable) + ``summary.json`` per view and a
``summary_by_view.json`` index.

This is a standalone eval — it does NOT touch the training pipeline or GripperheadFDMDataset.

    PYTHONPATH=. torchrun --nproc_per_node=1 examples/eval_gripperhead_fdm_rollout.py \
        --checkpoint <DCP iter_*/ dir> --base-config-dir <Cosmos3-Nano DCP dir> \
        --eval-set <dir of episode leaves> --tag <label>
"""
from cosmos_framework.inference.common.init import init_script

init_script()

# DATA-PARALLEL eval shards EPISODES across ranks (each rank runs the FULL model on its own subset),
# NOT model weights. init_script() already set this rank's CUDA device; now tear the process group back
# down so the model builds via the SAME path as single-GPU eval: parallel_dims=None -> parallelize_vfm_
# network skips fully_shard (no FSDP weight sharding) AND velocity_fn skips the per-step cross-rank CFG
# all_reduce. Without this, a rank that finishes its episode shard early stops issuing collectives and
# NCCL-deadlocks the ranks still sampling (deterministic hang at ~41/50). Coordination here is
# filesystem-only (.done_rank markers), so no process group is needed after device placement.
import torch.distributed as _dist
if _dist.is_initialized():
    _dist.destroy_process_group()

import argparse
import csv
import glob
import json
import math
import os
import pickle
import random
import time as _time
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.data.vfm.action.calib_segments import build_calib_segment_indices
from cosmos_framework.data.vfm.action.datasets.gripperhead_fdm_dataset import NEUTRAL_CAPTION_SEED
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.pose_utils import pose_abs_to_rel
from cosmos_framework.data.vfm.action.transforms import build_sequence_plan_from_mode
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import tokenize_caption

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAVE_SSIM = True
except Exception:
    _HAVE_SSIM = False

_RGB = ("agentview_rgb", "front_rgb", "base_camera_rgb", "render_camera_rgb", "left_rgb", "side_rgb", "rgb")
# Wrist / in-hand views — excluded from the auto-detected non-wrist camera set (still selectable explicitly).
_WRIST = ("wrist_rgb", "eye_in_hand_rgb", "hand_rgb", "robot0_eye_in_hand_rgb", "wrist", "eye_in_hand", "hand")


def _is_rgb_view(name: str) -> bool:
    """A frame-view dir name that looks like an RGB camera (``rgb`` or ``*_rgb``)."""
    return name == "rgb" or name.endswith("_rgb")


def _list_views(leaf: str, non_wrist_only: bool = True) -> list[str]:
    """RGB view subdirs (each holding ``video.mp4``) directly under ``leaf``. Wrist/in-hand views are
    dropped when ``non_wrist_only``. Ordered by the ``_RGB`` priority list, then alphabetically."""
    if not os.path.isdir(leaf):
        return []
    out = []
    for name in os.listdir(leaf):
        if not (_is_rgb_view(name) and os.path.isfile(os.path.join(leaf, name, "video.mp4"))):
            continue
        if non_wrist_only and name in _WRIST:
            continue
        out.append(name)
    pri = {v: i for i, v in enumerate(_RGB)}
    return sorted(out, key=lambda v: (pri.get(v, 999), v))


def discover_leaves(root: str, calib_dirname: str = "calibration") -> list[str]:
    """Recursively find episode leaves under ``root`` — a leaf is a dir with ``pose.pkl`` + >=1 RGB view
    subdir. Handles BOTH the flat ``ep_*/`` layout (each leaf directly under root) AND the nested
    training/format tree ``<task>/<episode_N>/<camera_poses_M>/{expert,counterfactual_*}/`` (maniskill,
    real_new, libero_annotated). ``calibration`` dirs are skipped — they are conditioning clips, not
    prediction targets. Sorted for a deterministic episode order."""
    leaves = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == calib_dirname:
            dirnames[:] = []  # never descend into (or evaluate) a calibration clip
            continue
        if ("pose.pkl" in filenames or "actions.pkl" in filenames) and _list_views(dirpath, non_wrist_only=False):
            leaves.append(dirpath)  # actions.pkl = native-action leaf (pose reconstructed in _load_pose)
    return sorted(leaves)


_REWRITES = [
    ("cosmos3._src.vfm.configs.base.", "cosmos_framework.configs.base."),
    ("cosmos3._src.vfm.models.", "cosmos_framework.model.vfm."),
    ("cosmos3._src.vfm.tokenizers.", "cosmos_framework.model.vfm.tokenizers."),
    ("cosmos3._src.imaginaire.", "cosmos_framework."),
]


# ------------------------------------------------------------------ model loading
def _config_from_dir(d: Path, vae_path: str = "") -> Cosmos3OmniConfig:
    cfg = d / "config.json" if (d / "config.json").exists() else d / "model" / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"no config.json under {d}")
    text = cfg.read_text()
    for a, b in _REWRITES:
        text = text.replace(a, b)
    model_cfg = json.loads(text)["model"]
    # VAE: force LOCAL loading from an absolute Wan2.2_VAE.pth when given. The base config ships
    # vae_path="pretrained/..." + bucket_name="bucket" => s3://bucket/..., which needs object-store
    # credentials; clearing bucket_name and pointing vae_path at the local file makes the eval
    # self-contained (same Wan2.2 VAE weights either way).
    wan = (vae_path or "").strip()
    if wan:
        def _override_vae(o):
            if isinstance(o, dict):
                if "vae_path" in o:            # the VIDEO tokenizer dict ("avae_path" is a different key)
                    o["vae_path"] = wan
                    o["bucket_name"] = ""
                for v in o.values():
                    _override_vae(v)
            elif isinstance(o, list):
                for v in o:
                    _override_vae(v)
        _override_vae(model_cfg)
        print(f"[eval] --vae-path override -> vae_path={wan} (bucket_name cleared)", flush=True)
    # Sound is DISABLED in gripperhead FDM training ({"override /sound_tokenizer": None} in the experiment
    # config), so the checkpoint has no sound expert and FDM eval is vision+action only. The base
    # config.json keeps sound_gen=True + a sound_tokenizer, which makes the model build fetch the audio
    # tokenizer (nvidia/Cosmos3-Nano/sound_tokenizer) — impossible without network. Disable it to match
    # how the checkpoint was trained (the sound branch is unused by FDM generation either way).
    try:
        if isinstance(model_cfg.get("config"), dict):
            model_cfg["config"]["sound_gen"] = False
            model_cfg["config"]["sound_tokenizer"] = None
            print("[eval] sound disabled (sound_gen=False, sound_tokenizer=None) — matches training", flush=True)
    except (KeyError, TypeError) as e:
        print(f"[eval] WARN: could not disable sound ({e!r})", flush=True)
    return Cosmos3OmniConfig(model=model_cfg)


def _patch_offline_vlm_processor(qwen_assets: str = ""):
    """Serve the VLM processor from a local dir so the eval can run offline.

    The VLM processor config (build_processor_lazy repository=nvidia/Cosmos3-Nano) otherwise runs
    `uvx hf@X download nvidia/Cosmos3-Nano`, which needs pypi + HuggingFace network. When
    ``qwen_assets`` points at a local qwen3vl_assets dir, monkeypatch the single download choke point
    (checkpoint_db._hf_download) to return that dir for the Cosmos3-Nano repo — build_processor then
    from_pretrained's the local files. No-op when unset (normal online download)."""
    qwen = (qwen_assets or "").strip()
    if not (qwen and os.path.isdir(qwen)):
        return
    import cosmos_framework.utils.checkpoint_db as _ckdb
    _orig = _ckdb._hf_download

    def _patched(cmd_args):
        repo = str(cmd_args[0]) if cmd_args else ""
        if "Cosmos3-Nano" in repo:
            print(f"[eval] offline VLM processor: _hf_download({repo}) -> local {qwen}", flush=True)
            return qwen
        return _orig(cmd_args)

    _ckdb._hf_download = _patched
    print(f"[eval] --qwen-assets -> patched _hf_download to serve Cosmos3-Nano from {qwen}", flush=True)


def _load_dcp_partial(weights: Path, config: Cosmos3OmniConfig, compile_enabled: bool = False):
    """Load a raw training DCP that saved only the trainable MoT (not the frozen Qwen ViT), which is
    absent from the checkpoint. FDM generation conditions on VAE latents + action + text and does not
    invoke the ViT, so a PARTIAL load (skip missing keys, leave ViT at init) is correct and avoids the
    heavyweight export_model consolidation."""
    import attrs
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig

    config.parallelism = attrs.asdict(ParallelismConfig())
    config.compile = attrs.asdict(CompileConfig(enabled=compile_enabled))
    model = Cosmos3OmniModel(config)
    sd = get_model_state_dict(model.model)
    # no_dist=True: single-process load path. Under an initialized 1-rank process group (torchrun or
    # init_script), some torch builds hit a DCP reduce_scatter bug ("metadata is None"); single-GPU eval
    # needs no cross-rank coordination, so bypass it.
    dcp.load(state_dict=sd, storage_reader=FileSystemReader(str(weights)),
             planner=DefaultLoadPlanner(allow_partial_load=True), no_dist=True)
    return model


def load_model(checkpoint: str, base_config_dir: str | None, vae_path: str = "",
               qwen_assets: str = "", compile_enabled: bool = False):
    _patch_offline_vlm_processor(qwen_assets)   # no-op unless --qwen-assets points at a local dir
    ckpt = Path(checkpoint)
    has_cfg = (ckpt / "config.json").exists() or (ckpt / "model" / "config.json").exists()
    if has_cfg:  # an exported/consolidated dir (config.json + full weights incl ViT) -> strict load
        wrapper = Cosmos3OmniModel.from_pretrained_dcp(
            ckpt, config=_config_from_dir(ckpt, vae_path), compile_config=CompileConfig(enabled=False))
        m = wrapper.model
    else:  # a raw training DCP iter dir -> partial load (frozen ViT not in the checkpoint)
        if not base_config_dir:
            raise FileNotFoundError(f"{checkpoint} has no config.json; pass --base-config-dir")
        config = _config_from_dir(Path(base_config_dir), vae_path)
        weights = ckpt / "model" if (ckpt / "model").exists() else ckpt
        m = _load_dcp_partial(Path(weights), config, compile_enabled).model
    # DATA-PARALLEL eval shards DIFFERENT episodes onto each rank, so ranks issue different numbers of
    # sampler (velocity_fn) calls. The model's per-step cross-rank collectives — the dp_shard CFG-align
    # all_reduce (omni_mot_model.py ~2508) plus cp/cfgp seed broadcasts — assume every rank steps in
    # lockstep on the SAME batch; here they desync and NCCL-timeout the moment the fastest rank finishes
    # (job aborts at ~41/50 every time). Weights load REPLICATED (no_dist), so each rank is fully
    # self-contained: null parallel_dims to force every collective site to its local no-op branch.
    if getattr(m, "parallel_dims", None) is not None:
        m.parallel_dims = None
    m.eval()
    return m


# ------------------------------------------------------------------ episode reading (eval_set: flat ep_* leaves)
def _read_video(path: str) -> np.ndarray:
    import imageio.v3 as iio
    try:
        return iio.imread(path, plugin="pyav")  # (T,H,W,C) uint8
    except Exception:
        return iio.imread(path)


def _resolve(leaf: str, cands) -> str | None:
    for c in cands:
        p = os.path.join(leaf, c, "video.mp4")
        if os.path.isfile(p):
            return p
    return None


def _load_pose(leaf: str, thr: float = 0.6):
    pp = os.path.join(leaf, "pose.pkl")
    if os.path.isfile(pp):
        d = pickle.load(open(pp, "rb"))
        mats = np.asarray(d["gripper_matrix"], dtype=np.float32)  # (T,4,4)
        gopen = np.asarray(d.get("gripper_open", np.ones(len(mats))), dtype=np.float32)
    else:
        # NATIVE-ACTION fallback: no realized pose.pkl -> reconstruct absolute poses from actions.pkl.
        # actions.pkl["actions"] is (N,7) body-frame [cm, euler_xyz_deg, open] (body_frame_osc_goal_delta).
        # integrate_action7 yields (N+1,4,4) with pose[t+1]=pose[t]@delta(act7[t]); pose_abs_to_rel's
        # backward_framewise (T_i^{-1}@T_{i+1}) is its EXACT inverse, so the eval recovers act7 verbatim
        # in whatever rep the run uses (cm+deg / raw m+rad -> meanstd). NOTE: these are OSC GOAL deltas,
        # not realized eef poses (~21% exceed the 5cm/step controller cap -> overshoot the true motion).
        ap = os.path.join(leaf, "actions.pkl")
        if not os.path.isfile(ap):
            raise FileNotFoundError(f"neither pose.pkl nor actions.pkl in {leaf}")
        from cosmos_framework.data.vfm.action.robocasa_native_action import integrate_action7
        a = pickle.load(open(ap, "rb"))
        act7 = np.asarray(a["actions"], dtype=np.float32)          # (N,7)
        mats = integrate_action7(act7).astype(np.float32)          # (N+1,4,4), pose0 = identity
        g0 = float(a.get("initial_gripper_open", 1.0))
        gopen = np.concatenate([[g0], act7[:, 6]]).astype(np.float32)  # (N+1,), 1=open
    gopen = (gopen > thr).astype(np.float32)
    return mats, gopen


def _build_action(mats: np.ndarray, gopen: np.ndarray,
                  convention: str = "backward_framewise",
                  rot_format: str = "euler_xyz",
                  trans_scale: float = 100.0, rot_scale: float = 57.2958) -> torch.Tensor:
    """abs poses -> per-step delta actions [Δpos(3), Δrot(rot_dim), gripper(1)].

    ALL of convention / rot_format / trans_scale / rot_scale MUST match training
    ([gripperhead].action_*). The shipped recipe is euler_xyz + trans_scale=100 +
    rot_scale=57.2958 => 7D [Δpos_cm, Δeuler_deg, gripper]."""
    poses_rel = pose_abs_to_rel(mats, rotation_format=rot_format, pose_convention=convention,
                                translation_scale=trans_scale, rotation_scale=rot_scale)
    grip = gopen[1:].reshape(-1, 1)
    return torch.from_numpy(np.concatenate([poses_rel, grip], axis=-1).astype(np.float32))


def _wrist_video_path(leaf: str):
    """Path to the leaf's wrist / in-hand view video (first present of ``_WRIST``), or None."""
    for w in _WRIST:
        p = os.path.join(leaf, w, "video.mp4")
        if os.path.isfile(p):
            return p
    return None


def read_episode(leaf: str, view: str, res: int, device: str, use_wrist: bool = False):
    """Return (frames_pm1 (T,3,H,W) in [-1,1], mats (T,4,4), gopen (T,), caption) for camera ``view``
    (a dir name like ``agentview_rgb`` / ``left_rgb`` / ``rgb``). For use_wrist the frame is
    [main | wrist] side-by-side (2:1 wide) using the leaf's wrist/in-hand view (main LEFT, wrist RIGHT),
    matching the wide ``<res>_w2`` wrist training tier; a missing wrist video falls back to a BLACK right
    half (mirrors the training ``_build_view`` fallback)."""
    rgb_p = os.path.join(leaf, view, "video.mp4")
    if not os.path.isfile(rgb_p):
        raise FileNotFoundError(f"no '{view}' video in {leaf}")
    rgb = _read_video(rgb_p)  # (T,H,W,3)
    T = len(rgb)
    if use_wrist:
        wp = _wrist_video_path(leaf)
        if wp is not None:
            wrist = _read_video(wp)  # (T,H',W',3)
            n = min(T, len(wrist))
            rgb, wrist = rgb[:n], wrist[:n]
            T = n
            if wrist.shape[1:3] != rgb.shape[1:3]:  # match main HxW (robocasa is already square-matched)
                wt = torch.from_numpy(wrist).float().permute(0, 3, 1, 2)
                wt = F.interpolate(wt, size=rgb.shape[1:3], mode="bilinear", align_corners=False)
                wrist = wt.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()
        else:
            wrist = np.zeros_like(rgb)  # black wrist half (matches training's missing-wrist fallback)
        frames = np.concatenate([rgb, wrist], axis=2)  # (T,H,2W,3) — main LEFT, wrist RIGHT
    else:
        frames = rgb
    v = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # (T,3,H,W)
    _, _, Hh, Ww = v.shape
    th = res
    tw = max(16, int(round((res * Ww / Hh) / 16) * 16))  # aspect-preserving, width snapped to /16 (VAE)
    if (Hh, Ww) != (th, tw):
        v = F.interpolate(v, size=(th, tw), mode="bilinear", align_corners=False)
    v = (v * 2 - 1).to(device)  # [-1,1]
    mats, gopen = _load_pose(leaf)
    n = min(T, len(mats))
    # FDM/IDM training always emits NEUTRAL_CAPTION_SEED
    # (GripperheadFDMDataset.force_neutral_caption), so resolving the episode's real ``lang`` here
    # would feed the model a prompt it never saw in training.
    caption = NEUTRAL_CAPTION_SEED
    return v[:n], mats[:n], gopen[:n], caption


# ------------------------------------------------------------------ batch
def build_fdm_batch(model, video_pm1, action, caption, num_history_frames, device, fps=15.0):
    # Forward dynamics: condition on the first ncl video latents + ALL actions, and generate the
    # remaining video latents. build_sequence_plan_from_mode turns that into the token-level plan.
    _, _, t, h, w = video_pm1.shape
    raw_dim = action.shape[1]
    a = torch.zeros(action.shape[0], model.config.max_action_dim, device=device)
    a[:, :raw_dim] = action.to(device)
    ncl = 1 + (max(1, num_history_frames) - 1) // 4
    sp = build_sequence_plan_from_mode("forward_dynamics", video_length=t, action_length=a.shape[0],
                                       has_text=True, num_condition_latent_frames=ncl)
    ids = tokenize_caption(caption, model.vlm_tokenizer, is_video=False,
                           use_system_prompt=model.vlm_config.use_system_prompt)
    return {
        model.input_video_key:   [video_pm1.to(device)],
        "action":                [a],
        "raw_action_dim":        [torch.tensor(raw_dim, dtype=torch.long, device=device)],
        "mode":                  [mode],
        model.input_caption_key: [caption],
        "text_token_ids":        [torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)],
        "image_size":            [torch.tensor([[h, w, h, w]], dtype=torch.float32, device=device)],
        # conditioning_fps drives the model's fps-modulation and MUST match training (the gripperhead
        # dataset feeds self.fps=15); 24 here silently degraded generation.
        "fps":                   torch.tensor([float(fps)], device=device),
        "conditioning_fps":      torch.tensor([float(fps)], device=device),
        "num_frames":            torch.tensor([t], device=device),
        "domain_id":             [torch.tensor(get_domain_id("gripperhead"), dtype=torch.long, device=device)],
        "sequence_plan":         [sp],
        "is_preprocessed":       True,
    }


def _to_u8(x: torch.Tensor) -> torch.Tensor:
    """(C,T,H,W) in [-1,1] -> (C,T,H,W) uint8 [0,255] (the multi-item VAE path expects uint8 items)."""
    return ((x.clamp(-1, 1) + 1) / 2 * 255.0).round().clamp(0, 255).to(torch.uint8)


def build_fdm_batch_multiitem(model, video_u8_list, action_list, caption, device, fps=15.0,
                              mode: str = "forward_dynamics"):
    """N-item FDM batch matching training's multi-item ``[calib x K?] + [history?] + [current+future]``.
    Every item EXCEPT the last is fully conditioning; the last (current+future) item conditions on its
    current-frame latent and generates the P future. Video items are uint8 (C,T,H,W) with NO
    is_preprocessed — the multi-item intake normalizes uint8->[-1,1] and re-stacks each to (1,C,T,H,W)
    for the VAE. ONE sequence_plan is built from the cf (last) item (ncl=1); the packer marks all
    non-last items fully conditioning and anchors each action block to its OWN item's temporal offset
    (share_vision_temporal_positions=False). Mirrors ActionTransformPipeline._call_multiitem."""
    maxD = model.config.max_action_dim

    def pad(a):
        p = torch.zeros(a.shape[0], maxD, device=device)
        p[:, : a.shape[1]] = a.to(device)
        return p

    vids = [v.to(device) for v in video_u8_list]
    acts = [pad(a) for a in action_list]
    dims = [torch.tensor(int(a.shape[1]), dtype=torch.long, device=device) for a in action_list]
    sizes = [torch.tensor([[v.shape[-2], v.shape[-1], v.shape[-2], v.shape[-1]]],
                          dtype=torch.float32, device=device) for v in vids]
    cf = vids[-1]
    sp = build_sequence_plan_from_mode(mode, video_length=cf.shape[1],
                                       action_length=action_list[-1].shape[0], has_text=True,
                                       num_condition_latent_frames=1)
    sp.share_vision_temporal_positions = False   # distinct time states per item — MUST match the transform
    ids = tokenize_caption(caption, model.vlm_tokenizer, is_video=False,
                           use_system_prompt=model.vlm_config.use_system_prompt)
    return {
        model.input_video_key:   [vids],   # per-sample list of N items
        "action":                [acts],    # per-sample list of N blocks
        "raw_action_dim":        [dims],    # per-sample list of N scalars
        "image_size":            [sizes],   # per-sample list of N image sizes
        "mode":                  [mode],
        model.input_caption_key: [caption],
        "text_token_ids":        [torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)],
        "fps":                   torch.tensor([float(fps)], device=device),
        "conditioning_fps":      torch.tensor([float(fps)], device=device),
        "domain_id":             [torch.tensor(get_domain_id("gripperhead"), dtype=torch.long, device=device)],
        "sequence_plan":         [sp],
        # NO is_preprocessed -> uint8 items get normalized + re-stacked to (1,C,T,H,W) for encode
    }


def build_fdm_batch_history_item(model, hist_u8, cf_u8, hist_act, cf_act, caption, device, fps=15.0):
    """Two-item (history + current+future) batch — the K=0 special case of build_fdm_batch_multiitem."""
    return build_fdm_batch_multiitem(model, [hist_u8, cf_u8], [hist_act, cf_act], caption, device, fps)


# ------------------------------------------------------------------ calibration items (mirrors training)
def resolve_calib_dir(leaf: str, calib_source_root: str | None = None, calib_dirname: str = "calibration",
                      view: str = "rgb"):
    """Find an episode's calibration dir for camera ``view``. Mirrors the dataset's _resolve_calib_dir
    (``<leaf>/calibration`` then the parent's — the parent case covers the nested
    ``<camera_poses_M>/{expert,counterfactual_*}`` layout where calibration is a sibling of the leaf),
    plus an optional ``--calib-source-root`` fallback for flat ``ep_*`` leaves that carry no inline
    calibration: parse the leaf name ``<idx>__<suite>__<task_demo>__<Manipulation>__<episode_N>__
    <camera_poses_M>__...`` to ``<root>/<suite>__<task_demo>/<Manipulation>/<episode_N>/<camera_poses_M>/calibration``."""
    def _ok(c):
        return (os.path.isdir(c) and os.path.isfile(os.path.join(c, view, "video.mp4"))
                and os.path.isfile(os.path.join(c, "pose.pkl")))
    for c in (os.path.join(leaf, calib_dirname),
              os.path.join(os.path.dirname(leaf.rstrip("/")), calib_dirname)):
        if _ok(c):
            return c
    if calib_source_root:
        name = os.path.basename(leaf.rstrip("/"))
        # leaf names look like "ep_000__<suite>__<task>__..." or "seg_000__..."; strip the "ep_NNN__" prefix
        body = name.split("__", 1)[1] if name.split("_", 1)[0] in ("ep", "seg") else name
        p = body.split("__")
        if len(p) >= 5:
            suite, task, manip, epi, cams = p[0], p[1], p[2], p[3], p[4]
            c = os.path.join(calib_source_root, f"{suite}__{task}", manip, epi, cams, calib_dirname)
            if _ok(c):
                return c
    return None


_ROT_DIMS = {"rot6d": 6, "euler_xyz": 3, "quat_xyzw": 4, "axisangle": 3, "rot9d": 9}


def build_null_calib_items(args, device, ep_frames):
    """Synthesize the K NULL calibration items for an episode that ships NO ``calibration/`` clip.

    Mirrors the dataset's p_include_calibration<1 path VERBATIM (gripperhead_fdm_dataset
    ``_load_sample_multiitem``): the K slots are KEPT but their content is zeroed — black uint8 video
    (C, calib_seg_len, h, w) + an all-zero (calib_seg_len-1, D) action block. So a calibmix checkpoint sees
    exactly the "calibration dropped" case it trained on, instead of the episode being skipped (which is
    what happens when --use-calibration is set and no calib dir resolves) or the slots vanishing entirely
    (which the model never saw). Requires --calib-null."""
    C, h, w = int(ep_frames.shape[1]), int(ep_frames.shape[2]), int(ep_frames.shape[3])
    L = int(args.calib_seg_len)
    D = 3 + _ROT_DIMS.get(args.action_rot_format, 6) + 1
    vids = [torch.zeros((C, L, h, w), dtype=torch.uint8, device=device) for _ in range(int(args.calib_segments))]
    acts = [torch.zeros((max(1, L) - 1, D), dtype=torch.float32) for _ in range(int(args.calib_segments))]
    return vids, acts


def build_calib_eval_items(calib_dir: str, args, device, view: str = "rgb"):
    """Load + segment the calibration clip (camera ``view``) into (K uint8 video items, K action blocks),
    matching the dataset's _build_calib_items. Frames are resized/normalized EXACTLY like read_episode (so
    calib items share the cf/history resolution), then uint8 for the multi-item VAE intake. Segments come
    from build_calib_segment_indices on the subsampled calib poses (move_range signs); actions use the SAME
    convention/rot_format/scales/norm as the episode. No axis aug at eval."""
    rgb_p = os.path.join(calib_dir, view, "video.mp4")
    if not os.path.isfile(rgb_p):
        raise FileNotFoundError(f"no calib '{view}' video in {calib_dir}")
    frames = _read_video(rgb_p)
    T = len(frames)
    v = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # (T,C,H,W)
    _, _, Hh, Ww = v.shape
    th, tw = args.resolution, max(16, int(round((args.resolution * Ww / Hh) / 16) * 16))
    if (Hh, Ww) != (th, tw):
        v = F.interpolate(v, size=(th, tw), mode="bilinear", align_corners=False)
    v = (v * 2 - 1)                                                   # (T,C,th,tw) [-1,1]
    mats, gopen = _load_pose(calib_dir)                              # calib_dir/pose.pkl
    n = min(len(mats), T)
    move_order = None
    mrp = os.path.join(calib_dir, "move_range.pkl")
    if os.path.isfile(mrp):
        try:
            move_order = pickle.load(open(mrp, "rb")).get("movement_order")
        except Exception:
            move_order = None
    sub = list(range(0, n, args.calib_frame_interval)) or [0]        # subsample before segment detection
    efficient = (args.calib_segments == 6)                           # 6 -> per-DoF (move_order signs); 12 -> both signs
    seg_lists = build_calib_segment_indices(mats[sub], args.calib_seg_len, efficient, move_order)
    vids, acts = [], []
    for seg in seg_lists:
        real = [sub[i] for i in seg]                                 # map subsampled -> raw calib frame idx
        item = _to_u8(v[real].permute(1, 0, 2, 3)).to(device)        # (C,seg_len,th,tw) uint8
        act = _build_action(mats[real], gopen[real], args.action_convention,
                            args.action_rot_format, args.action_trans_scale, args.action_rot_scale)  # (seg_len-1,D)
        vids.append(item)
        acts.append(act)
    if getattr(args, "calib_null", False):
        # In-distribution NULL-calibration: training keeps the K calib slots but ZEROS their content
        # (dataset __getitem__ p_include_calibration<1 path: black video + zero action, no separate flag).
        # Keep count+shape, zero the content -> the model sees the exact "calib present but padded/dropped"
        # case it trained on. Differs from --use-calibration off, which OMITS the slots entirely; history
        # stays forced because the slots remain and encode_exact_durations still includes calib_seg_len.
        vids = [torch.zeros_like(x) for x in vids]
        acts = [torch.zeros_like(a) for a in acts]
    return vids, acts


# ------------------------------------------------------------------ metrics
# Metric helpers ported VERBATIM from robotics-wan-dev/inference_rlbench_calib_context_segments_full_episode.py
# (compute_psnr / compute_ssim / _gaussian_window) so the metrics_full_episode.csv values are directly
# comparable to robotics-wan's eval_results. PSNR on uint8 MSE; SSIM = Gaussian-window (11,1.5), C1=0.01^2,
# C2=0.03^2; LPIPS = AlexNet on [-1,1]. All per-frame, then mean over the segment.
def _to_tensor_01(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    g = torch.exp(-((coords - (window_size - 1) / 2.0) ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    g2 = torch.outer(g, g)
    g2 = g2 / g2.sum()
    return g2.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def _rw_psnr(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    diff = pred_u8.astype(np.float32) - gt_u8.astype(np.float32)
    mse = float(np.mean(diff * diff))
    return float("inf") if mse <= 1e-10 else float(10.0 * math.log10(255.0 * 255.0 / mse))


def _rw_ssim(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    x, y = _to_tensor_01(pred_u8), _to_tensor_01(gt_u8)
    c = x.shape[1]
    win = _gaussian_window(11, 1.5, c)
    pad = 11 // 2
    mux = F.conv2d(x, win, padding=pad, groups=c)
    muy = F.conv2d(y, win, padding=pad, groups=c)
    mxy, mx2, my2 = mux * muy, mux.pow(2), muy.pow(2)
    sx2 = F.conv2d(x * x, win, padding=pad, groups=c) - mx2
    sy2 = F.conv2d(y * y, win, padding=pad, groups=c) - my2
    sxy = F.conv2d(x * y, win, padding=pad, groups=c) - mxy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mxy + c1) * (2 * sxy + c2)) / (((mx2 + my2 + c1) * (sx2 + sy2 + c2)) + 1e-12)
    return float(ssim_map.mean().item())


def _lpips_u8(lpips_model, pred_u8: np.ndarray, gt_u8: np.ndarray, device) -> float:
    def t(a):
        return (torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1).to(device)
    with torch.no_grad():
        return float(lpips_model(t(pred_u8), t(gt_u8)).item())


def _seg_metrics_rw(pred_u8: np.ndarray, gt_u8: np.ndarray, lpips_model, device) -> dict:
    """pred/gt: (T,H,W,C) uint8. Per-frame psnr/ssim/lpips averaged over the segment.
    num_frames_eval = min(len(pred), len(gt)). PSNR/SSIM are ported verbatim from robotics-wan so the
    values stay directly comparable with its eval_results."""
    n = min(len(pred_u8), len(gt_u8))
    ps, ss = [], []
    lp = [] if lpips_model is not None else None
    for i in range(n):
        p, g = pred_u8[i], gt_u8[i]
        if p.shape != g.shape:
            p = np.array(Image.fromarray(p).resize((g.shape[1], g.shape[0]), resample=Image.BICUBIC))
        ps.append(_rw_psnr(p, g))
        ss.append(_rw_ssim(p, g))
        if lp is not None:
            lp.append(_lpips_u8(lpips_model, p, g, device))
    mean = lambda values: float(np.mean(values)) if len(values) else float("nan")
    return {"num_frames_eval": n, "psnr": mean(ps), "ssim": mean(ss),
            "lpips": mean(lp) if lp is not None else float("nan")}


# ------------------------------------------------------------------ robotics-wan-style rollout
def _gen_window(model, ep_frames, gen_frames, mats, gopen, caption, calib_v, calib_a, start, args, device, custom_act=None):
    """Generate ONE window at frame ``start``: current = gen_frames[start] (GT at rollout round 0,
    model-generated after write-back), history from gen_frames, GT-for-metrics from ep_frames. Dispatches
    calib / history / single. Returns (pred_u8 (P+1,H,W,C), gt_u8 (P+1,H,W,C)) uint8 — the full decoded
    current+future window (incl the conditioning current frame, matching robotics-wan's sample window)."""
    T = ep_frames.shape[0]
    H, P, S = args.num_history_frames, args.num_pred_frames, args.history_frame_stride
    clamp = lambda x: max(0, min(int(x), T - 1))
    use_history, use_calib = H > 1, args.use_calibration
    cf_idx = [clamp(start)] + [clamp(start + 1 + j) for j in range(P)]           # current + P future (clamped)
    cf_frames = torch.stack([gen_frames[clamp(start)]]
                            + [ep_frames[clamp(start + 1 + j)] for j in range(P)], dim=1)   # (C,P+1,h,w)
    # custom-action mode: drive the window with a synthesized cm/deg action instead of the GT-derived one
    # (the history action block below stays REAL — it is observed context, not the commanded motion).
    cf_act = (custom_act.to(torch.float32) if custom_act is not None else
              _build_action(mats[cf_idx], gopen[cf_idx], args.action_convention,
                            args.action_rot_format, args.action_trans_scale, args.action_rot_scale))
    if use_calib or use_history:
        items_v = list(calib_v) if use_calib else []
        items_a = list(calib_a) if use_calib else []
        if use_history:
            hist_idx = [clamp(start - S * (H - i)) for i in range(H)]           # H sparse, before current
            hist_frames = torch.stack([gen_frames[i] for i in hist_idx], dim=1)
            hist_act = _build_action(mats[hist_idx], gopen[hist_idx], args.action_convention,
                                     args.action_rot_format, args.action_trans_scale, args.action_rot_scale)
            items_v.append(_to_u8(hist_frames)); items_a.append(hist_act)
        items_v.append(_to_u8(cf_frames)); items_a.append(cf_act)             # cf item LAST (the generated one)
        batch = build_fdm_batch_multiitem(model, items_v, items_a, caption, device, fps=args.fps)
    else:
        batch = build_fdm_batch(model, cf_frames.unsqueeze(0), cf_act, caption, 1, device, fps=args.fps)
    with torch.no_grad():
        outputs = model.generate_samples_from_batch(
            batch, guidance=1.0, action_guidance=args.action_cfg_scale, seed=[args.seed], num_steps=args.num_steps)
        dec = model.decode(outputs["vision"][0])[0].clamp(-1, 1)             # (C,P+1,h,w) — cf item
    pred_u8 = (((dec.float().permute(1, 2, 3, 0) + 1) / 2 * 255).round().clamp(0, 255).byte().cpu().numpy())  # (P+1,H,W,C)
    gt_win = torch.stack([ep_frames[clamp(start + i)] for i in range(P + 1)], dim=0)  # (P+1,C,h,w) clamped
    gt_u8 = (((gt_win.float().permute(0, 2, 3, 1) + 1) / 2 * 255).round().clamp(0, 255).byte().cpu().numpy())
    return pred_u8, gt_u8


def rollout_episode_rw(model, ep_frames, mats, gopen, caption, calib_v, calib_a, args, device):
    """robotics-wan-style rollout (matches inference_rlbench_calib_context_segments_full_episode.py):
    for each segment start s = seg_idx*stride (stride = num_pred_frames), run ``num_rollout_rounds``
    autoregressive rounds — round 0 teacher-forced (history=GT), each later round feeds the model's OWN
    generated frames back as history (closed-loop write-back) — and stitch the rounds (dropping the
    1-frame overlap) into ONE (pred, gt) pair per segment. Returns list of
    (seg_idx, start_frame, pred_u8 (T,H,W,C), gt_u8 (T,H,W,C))."""
    T = ep_frames.shape[0]
    P = args.num_pred_frames
    stride = max(1, P)                       # robotics-wan stride = sampled_length-1 = num_pred_frames
    rounds = max(1, int(args.num_rollout_rounds))
    num_segments = max(1, math.ceil(T / float(stride)))
    out_segs = []
    for seg_idx in range(num_segments):
        roll_start = seg_idx * stride
        if roll_start >= T - 1:              # no real future frame to predict
            break
        gen_frames = ep_frames.clone()       # fresh per segment -> round 0 is teacher-forced (history=GT)
        pred_acc, gt_acc = [], []
        for r in range(rounds):
            start = roll_start + r * stride
            pred_u8, gt_u8 = _gen_window(model, ep_frames, gen_frames, mats, gopen, caption,
                                         calib_v, calib_a, start, args, device)
            if rounds > 1:                   # write generated frames back for the next closed-loop round
                for i in range(min(pred_u8.shape[0], gen_frames.shape[0] - start)):
                    fr = pred_u8[i].astype(np.float32) / 255.0 * 2.0 - 1.0
                    gen_frames[start + i] = torch.from_numpy(fr).permute(2, 0, 1).to(gen_frames)
            skip = 0 if r == 0 else 1        # drop the 1-frame overlap between consecutive rounds
            m = min(pred_u8.shape[0], gt_u8.shape[0])
            pred_acc.extend(pred_u8[i] for i in range(skip, m))
            gt_acc.extend(gt_u8[i] for i in range(skip, m))
        if pred_acc:
            out_segs.append((seg_idx, roll_start, np.stack(pred_acc, 0), np.stack(gt_acc, 0)))
    return out_segs


def _save_sbs_video(path, pred_u8, gt_u8, fps=30):
    """[GT | pred] side-by-side (concat along width), padded to equal length (repeat last frame). For a
    mask run each frame is already [RGB|mask], so this yields [GT_rgb|GT_mask|pred_rgb|pred_mask]."""
    n = max(len(pred_u8), len(gt_u8))
    def _pad(a):
        return a if len(a) >= n else np.concatenate([a, np.repeat(a[-1:], n - len(a), axis=0)], axis=0)
    g, p = _pad(gt_u8), _pad(pred_u8)
    frames = [np.concatenate([g[i], p[i]], axis=1) for i in range(n)]
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)


# robotics-wan metrics_full_episode.csv schema (per-segment rows + a final 'average' row).
_SCHEMA = ["episode", "segment_index", "start_frame", "num_frames_eval", "psnr", "ssim", "lpips"]


def _run_view_eval(model, leaves, view, args, out_view_dir, lpips_model, device, rank=0, world=1):
    """Full rollout eval for ONE camera ``view``, DATA-PARALLEL across ``world`` GPUs on one node: rank r
    evaluates episodes ``leaves[r::world]`` on its own GPU (init_script set_device'd it), writing each
    episode's videos + a per-episode ``_metrics.json`` marker to the SHARED ``out_view_dir`` under the
    GLOBAL index ``episode_{k}``. Markers make it idempotent/resumable (skip already-done episodes across
    ranks AND across resubmits). After its shard each rank drops ``.done_rank{r}``; rank 0 then waits for
    all ranks (filesystem barrier w/ timeout) and aggregates ALL markers -> metrics_full_episode.csv +
    summary.json, returning the average-metrics dict. Non-zero ranks return None."""
    out_view_dir.mkdir(parents=True, exist_ok=True)
    n_calib_miss = n_done = n_resumed = n_calib_synth = 0
    for k in range(rank, len(leaves), world):            # this rank's shard (global episode index k)
        leaf = leaves[k]
        episode_name = f"episode_{k:04d}"
        episode_dir = out_view_dir / episode_name
        done_marker = episode_dir / "_metrics.json"      # present == episode fully done (resume/skip)
        if done_marker.exists():
            try:
                if json.loads(done_marker.read_text()):
                    n_resumed += 1
                    continue
            except Exception:
                pass  # corrupt/partial marker -> re-evaluate
        try:
            ep_frames, mats, gopen, caption = read_episode(leaf, view, args.resolution, device, use_wrist=args.use_wrist)
        except Exception as e:
            print(f"[{args.tag}/{view}][r{rank}] skip {episode_name} ({os.path.basename(leaf)[:30]}): {e!r}", flush=True)
            continue
        calib_v = calib_a = None
        if args.use_calibration:
            calib_dir = resolve_calib_dir(leaf, args.calib_source_root, args.calib_dirname, view)
            if calib_dir is None and args.calib_null:
                # No calibration clip in this set -> KEEP the K slots and zero them (in-distribution for a
                # calibmix ckpt). Without --calib-null this stays a skip, as before.
                calib_v, calib_a = build_null_calib_items(args, device, ep_frames)
                n_calib_synth += 1
            elif calib_dir is None:
                n_calib_miss += 1
                print(f"[{args.tag}/{view}][r{rank}] skip {episode_name}: no calib dir "
                      f"(--calib-source-root={args.calib_source_root}; pass --calib-null to run with "
                      f"ZEROED calib slots instead)", flush=True)
                continue
            else:
                calib_v, calib_a = build_calib_eval_items(calib_dir, args, device, view)
        segs = rollout_episode_rw(model, ep_frames, mats, gopen, caption, calib_v, calib_a, args, device)
        episode_dir.mkdir(parents=True, exist_ok=True)
        ep_rows, ep_psnr = [], []
        for (seg_idx, start_frame, pred_u8, gt_u8) in segs:
            m = _seg_metrics_rw(pred_u8, gt_u8, lpips_model, device)
            ep_rows.append({"episode": episode_name, "segment_index": seg_idx, "start_frame": start_frame, **m})
            if np.isfinite(m["psnr"]):
                ep_psnr.append(m["psnr"])
            if args.side_by_side:
                _save_sbs_video(str(episode_dir / f"seg{seg_idx:04d}_start{start_frame:06d}.mp4"), pred_u8, gt_u8)
        done_marker.write_text(json.dumps(ep_rows))       # after the videos -> "fully done"
        n_done += 1
        pm = f"{np.mean(ep_psnr):.2f}" if ep_psnr else "n/a"
        print(f"[{args.tag}/{view}][r{rank}] {episode_name} ({k+1}/{len(leaves)}) {os.path.basename(leaf)[:30]}: "
              f"segs={len(segs)} psnr~{pm}", flush=True)

    # ---- filesystem barrier: signal this rank done for this view, then only rank 0 aggregates ----
    (out_view_dir / f".done_rank{rank}").write_text("done")
    print(f"[{args.tag}/{view}][r{rank}] shard done: {n_done} new, {n_resumed} resumed, "
          f"{n_calib_miss} calib-miss, {n_calib_synth} null-calib(synthesized)", flush=True)
    if rank != 0:
        return None
    if world > 1:
        deadline = _time.time() + int(args.barrier_timeout)
        while _time.time() < deadline and not all((out_view_dir / f".done_rank{r}").exists() for r in range(world)):
            _time.sleep(10)
        missing = [r for r in range(world) if not (out_view_dir / f".done_rank{r}").exists()]
        if missing:
            print(f"[{args.tag}/{view}] WARNING: barrier timeout; ranks {missing} not done — aggregating partial", flush=True)

    # ---- rank 0: aggregate ALL per-episode markers (every rank's) into the final CSV + summary ----
    all_rows = []
    for md in sorted(out_view_dir.glob("episode_*/_metrics.json")):
        try:
            all_rows.extend(json.loads(md.read_text()))
        except Exception:
            pass
    if not all_rows:
        print(f"[{args.tag}/{view}] NO segments evaluated (calib_miss on r0={n_calib_miss}).", flush=True)
        return None

    def _col_avg(key):  # mean over FINITE values only (drop inf PSNR / nan), matches robotics-wan _col_avg
        vals = [r[key] for r in all_rows if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    with open(out_view_dir / "metrics_full_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SCHEMA); w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, float("nan")) for c in _SCHEMA})
        w.writerow({"episode": "average", "segment_index": "", "start_frame": "",
                    "num_frames_eval": int(np.mean([r["num_frames_eval"] for r in all_rows])),
                    **{c: _col_avg(c) for c in _SCHEMA[4:]}})

    avg = {c: _col_avg(c) for c in _SCHEMA[4:]}
    n_ep = len({r["episode"] for r in all_rows})
    summary = {"tag": args.tag, "camera_view": view, "checkpoint": args.checkpoint, "n_episodes": n_ep,
               "n_segments": len(all_rows), "num_rollout_rounds": args.num_rollout_rounds,
               "action_cfg_scale": args.action_cfg_scale, "world_size": world,
               "average_metrics": {c: (round(v, 5) if np.isfinite(v) else None) for c, v in avg.items()}}
    (out_view_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[{args.tag}/{view}] ===== AVERAGE over {n_ep} episodes / {len(all_rows)} segments (world={world}) =====", flush=True)
    for c, v in avg.items():
        print(f"    {c:12s} = {v:.4f}", flush=True)
    print(f"[{args.tag}/{view}] output: {out_view_dir}", flush=True)
    return avg


def main():
    ap = argparse.ArgumentParser(
        description="Regular full-episode evaluation for the gripperhead forward-dynamics recipe. "
                    "The conditioning defaults are the shipped training recipe and the sampling "
                    "defaults are a measured fast preset, so only --checkpoint, --eval-set and "
                    "--tag are normally needed.")

    g = ap.add_argument_group("model + I/O")
    g.add_argument("--checkpoint", required=True, help="DCP iter_*/ dir (or an exported model dir)")
    g.add_argument("--base-config-dir", default=None,
                   help="dir holding config.json; required for a raw training DCP (which stores no config)")
    g.add_argument("--eval-set", required=True,
                   help="dir of episode leaves, searched recursively. A leaf is any dir holding "
                        "<view>/video.mp4 + pose.pkl; calibration clips are excluded from the episode set.")
    g.add_argument("--tag", required=True, help="label for this run; results go to <out>/<tag>/<view>/")
    g.add_argument("--out", default="outputs/fdm_rollout_eval")
    g.add_argument("--vae-path", default=os.environ.get("WAN_VAE_PATH", ""),
                   help="absolute Wan2.2_VAE.pth. Without it the base config resolves the VAE through the "
                        "object store, which needs credentials. Defaults to $WAN_VAE_PATH.")
    g.add_argument("--qwen-assets", default=os.environ.get("QWEN3VL_ASSETS", ""),
                   help="local text-tokenizer/processor dir, so the run needs no HuggingFace network. "
                        "Defaults to $QWEN3VL_ASSETS (the same variable training uses).")

    g = ap.add_argument_group("episode + view selection")
    g.add_argument("--camera-views", default="",
                   help="comma-separated camera-view dir names to evaluate (e.g. 'rgb', 'agentview_rgb', "
                        "or 'left_rgb,side_rgb'). Each view is evaluated independently into "
                        "<out>/<tag>/<view>/. Empty => auto-detect every non-wrist RGB view in the eval set.")
    g.add_argument("--max-episodes", type=int, default=0, help="cap #episodes (0 = all)")
    g.add_argument("--calib-dirname", default="calibration",
                   help="calibration subdir name under a leaf (or its parent)")
    g.add_argument("--calib-source-root", default=None,
                   help="fallback root for leaves with no inline calibration; the calib dir is resolved by "
                        "parsing the leaf name against <root>/<task>/<episode_N>/<camera_poses_M>/<calib-dirname>")

    g = ap.add_argument_group("conditioning horizon (MUST match the trained checkpoint)")
    g.add_argument("--num-history-frames", type=int, default=25,
                   help="1 => plain current+future (one joint clip); >1 (must be 4n+1) => history is a "
                        "SEPARATE VAE item. Matches [gripperhead].num_history")
    g.add_argument("--num-pred-frames", type=int, default=16, help="matches [gripperhead].num_pred")
    g.add_argument("--history-frame-stride", type=int, default=3,
                   help="sparse-history stride; matches [gripperhead].history_stride")
    g.add_argument("--resolution", type=int, default=512,
                   help="square resolution tier; matches [gripperhead].resolution")
    g.add_argument("--fps", type=float, default=15.0, help="conditioning_fps for fps-modulation")
    g.add_argument("--use-wrist", action="store_true",
                   help="build the [main | wrist] side-by-side (2:1 wide) input matching the wrist training "
                        "tier ([gripperhead].use_wrist). A missing wrist video falls back to a black right "
                        "half. Not supported together with --use-calibration.")

    g = ap.add_argument_group("calibration context (MUST match the trained checkpoint)")
    g.add_argument("--use-calibration", action=argparse.BooleanOptionalAction, default=True,
                   help="prepend K per-DoF calibration segments as extra fully-conditioning VAE items, "
                        "giving [calib x K] (+ [history]) + [current+future]. On by default to match the "
                        "shipped recipe; pass --no-use-calibration for a checkpoint trained without it.")
    g.add_argument("--calib-null", action="store_true",
                   help="in-distribution NULL calibration: keep the K calib slots but ZERO their content, "
                        "matching training's p_include_calibration<1 path. Differs from "
                        "--no-use-calibration, which drops the slots entirely. Also lets an eval set with "
                        "no calibration clips run against a calibration-trained checkpoint.")
    g.add_argument("--calib-segments", type=int, default=6, choices=[6, 12],
                   help="K: 6 => per-DoF (uses move_range signs), 12 => both signs. "
                        "Matches [gripperhead].calib_segments")
    g.add_argument("--calib-seg-len", type=int, default=5,
                   help="frames per calib segment; matches [gripperhead].calib_seg_len")
    g.add_argument("--calib-frame-interval", type=int, default=3,
                   help="subsample stride on the raw calib clip before segment detection; "
                        "matches [gripperhead].calib_frame_interval")

    g = ap.add_argument_group("action representation (MUST match the trained checkpoint)")
    g.add_argument("--action-convention", default="backward_framewise",
                   choices=["backward_framewise", "backward_anchored"],
                   help="matches [gripperhead].action_pose_convention")
    g.add_argument("--action-rot-format", default="euler_xyz",
                   choices=["rot6d", "euler_xyz", "quat_xyzw", "axisangle", "rot9d"],
                   help="matches [gripperhead].action_rotation_format")
    g.add_argument("--action-trans-scale", type=float, default=100.0,
                   help="100 = m->cm; matches [gripperhead].action_translation_scale")
    g.add_argument("--action-rot-scale", type=float, default=57.2958,
                   help="57.2958 = rad->deg; matches [gripperhead].action_rotation_scale")

    # Defaults here are the FAST preset (~3x the reference sampler settings for -0.19 dB PSNR; see
    # the table in README "Evaluation"). To reproduce the reference numbers exactly, pass
    #   --num-steps 35 --action-cfg-scale 5.0 --no-compile
    g = ap.add_argument_group("sampling + rollout")
    g.add_argument("--action-cfg-scale", type=float, default=1.0,
                   help="classifier-free guidance scale on the action conditioning. 1.0 (default) "
                        "runs ONE model forward per step; any other value adds a null-action branch "
                        "and doubles the forwards. Raise to 5.0 to match a guidance-using deployment.")
    g.add_argument("--num-steps", type=int, default=20,
                   help="diffusion sampling steps per window. 20 (default) costs ~0.02 dB PSNR "
                        "against 35 and is 1.7x faster.")
    g.add_argument("--num-rollout-rounds", type=int, default=2,
                   help="autoregressive rounds per segment: round 0 is teacher-forced (history = GT), each "
                        "later round feeds the model's OWN frames back (closed-loop). 1 = pure teacher "
                        "forcing; 2 surfaces one step of error accumulation. This changes WHAT is "
                        "measured, so it is not a speed knob.")
    g.add_argument("--seed", type=int, default=0)

    g = ap.add_argument_group("output + runtime")
    g.add_argument("--no-lpips", action="store_true", help="skip LPIPS (leaves the column empty)")
    g.add_argument("--no-side-by-side", dest="side_by_side", action="store_false",
                   help="skip writing the per-segment [GT|pred] mp4s (metrics only)")
    g.add_argument("--compile", dest="compile_model", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="torch.compile the model: ~1.2x faster with no measurable metric change. "
                        "The first window pays the compile warmup, which a run of more than a few "
                        "segments already amortizes. --no-compile to disable.")
    g.add_argument("--barrier-timeout", type=int, default=1800,
                   help="seconds rank 0 waits for the other ranks' episode shards before aggregating "
                        "whatever finished (multi-GPU only)")
    ap.set_defaults(side_by_side=True)
    args = ap.parse_args()

    if args.use_wrist and args.use_calibration:
        ap.error("--use-wrist + --use-calibration is not supported (calib clips are not wrist-stitched); "
                 "pass --no-use-calibration for a wrist checkpoint")

    device = "cuda"   # init_script already ran torch.cuda.set_device(local_rank) when world>1 -> per-rank GPU
    # DATA-PARALLEL across the GPUs on ONE node (torchrun --nproc_per_node=world): rank r evaluates the
    # episode shard leaves[r::world] on its own GPU; rank 0 aggregates all ranks' per-episode markers at
    # the end. RANK/WORLD_SIZE come from torchrun; world==1 is plain single-GPU eval.
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # Per-view output layout: <out>/<tag>/<view>/episode_{idx:04d}/seg{seg:04d}_start{frame:06d}.mp4 +
    # metrics_full_episode.csv + summary.json (ONE subfolder per evaluated camera view).
    save_dir = Path(args.out) / args.tag
    save_dir.mkdir(parents=True, exist_ok=True)

    # discover_leaves recurses, so BOTH the flat ep_*/ layout AND the nested training-format tree
    # <task>/<episode_N>/<camera_poses_M>/{expert,counterfactual_*}/ are supported.
    leaves = discover_leaves(args.eval_set, args.calib_dirname)
    if args.max_episodes > 0:
        leaves = leaves[: args.max_episodes]

    # Camera views: explicit --camera-views, else auto-detect all NON-WRIST rgb views from the first leaf
    # that has any (each view is evaluated independently into <out>/<tag>/<view>/).
    if args.camera_views.strip():
        views = [v.strip() for v in args.camera_views.split(",") if v.strip()]
    else:
        views = next((vs for d in leaves for vs in [_list_views(d, non_wrist_only=True)] if vs), [])
        if not views:
            print(f"[{args.tag}] FATAL: no non-wrist RGB camera views auto-detected under "
                  f"{args.eval_set}; pass --camera-views explicitly. Aborting.", flush=True)
            raise SystemExit(3)

    # Fail fast if nothing resolves for ANY selected view (wrong eval-set or view names) — otherwise the
    # loop evaluates 0 episodes and exits 0, which reads as success.
    def _leaf_has_view(d):
        return os.path.isdir(d) and any(os.path.isfile(os.path.join(d, v, "video.mp4")) for v in views)
    n_valid = sum(1 for d in leaves if _leaf_has_view(d))
    if n_valid == 0:
        print(f"[{args.tag}] FATAL: 0 of {len(leaves)} episodes under {args.eval_set} have any of "
              f"views={views} — are the view names correct? Aborting.", flush=True)
        raise SystemExit(3)
    if n_valid < len(leaves):
        print(f"[{args.tag}] WARNING: only {n_valid}/{len(leaves)} episodes have >=1 of views={views}", flush=True)
    print(f"[{args.tag}] {len(leaves)} episodes | views={views} | "
          f"action_cfg={args.action_cfg_scale} rollout_rounds={args.num_rollout_rounds}", flush=True)

    lpips_model = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex").to(device).eval()
            print(f"[{args.tag}] LPIPS(alex) loaded", flush=True)
        except Exception as e:
            print(f"[{args.tag}] LPIPS unavailable ({e!r}); skipping", flush=True)

    model = load_model(args.checkpoint, args.base_config_dir, args.vae_path, args.qwen_assets,
                       args.compile_model)

    # Dispatch (mirrors the training dataset): use_calibration => [calib x K] (+ history) + current+future
    # multi-item; else num_history_frames>1 => [history | current+future] two-item; else single joint clip.
    use_history = args.num_history_frames > 1
    use_calib = args.use_calibration
    if use_calib or use_history:
        # Every separate VAE item (calib segments, history, current+future) must encode at its EXACT
        # length without padding, else per-item latent counts — and thus the sequence plan — mismatch.
        # Mirrors sft_config.gripperhead_derived_overrides' encode_exact_durations.
        dur = {args.num_pred_frames + 1}
        if use_history:
            dur.add(args.num_history_frames)
        if use_calib:
            dur.add(args.calib_seg_len)
        dur = sorted(dur)
        tok = model.tokenizer_vision_gen
        tok.encode_exact_durations = list(dur)
        if hasattr(tok, "model") and hasattr(tok.model, "_encode_exact_durations"):
            tok.model._encode_exact_durations = set(dur)
        print(f"[{args.tag}] multi-item (calib={use_calib} history={use_history}): "
              f"encode_exact_durations={dur}", flush=True)

    # Evaluate each selected camera view independently, into its own <out>/<tag>/<view>/ subfolder.
    # Each view is data-parallel across ranks (rank r does leaves[r::world]); rank 0 aggregates.
    combined = {}
    for view in views:
        vleaves = [d for d in leaves if os.path.isfile(os.path.join(d, view, "video.mp4"))]
        if not vleaves:
            if rank == 0:
                print(f"[{args.tag}/{view}] no episodes have this view; skipping", flush=True)
            continue
        if len(vleaves) < len(leaves) and rank == 0:
            print(f"[{args.tag}/{view}] {len(vleaves)}/{len(leaves)} episodes have this view", flush=True)
        avg = _run_view_eval(model, vleaves, view, args, save_dir / view, lpips_model, device, rank, world)
        if avg is not None:   # only rank 0 aggregates + returns metrics
            combined[view] = avg

    # Non-zero ranks are done once their shards + per-view barriers complete; only rank 0 writes the
    # cross-view index and decides the exit code.
    if rank != 0:
        return
    if not combined:
        print(f"[{args.tag}] NO view produced any evaluated segments.", flush=True)
        raise SystemExit(2)

    # Cross-view index; per-view detail lives in each <view>/summary.json.
    (save_dir / "summary_by_view.json").write_text(json.dumps(
        {"tag": args.tag, "checkpoint": args.checkpoint, "views": views,
         "average_metrics_by_view": {v: {c: (round(x, 5) if x == x else None) for c, x in a.items()}
                                     for v, a in combined.items()}}, indent=2))
    print(f"\n[{args.tag}] ===== DONE: {len(combined)} view(s) =====", flush=True)
    for v, a in combined.items():
        ps, ss = a.get("psnr", float("nan")), a.get("ssim", float("nan"))
        line = f"    {v:16s} psnr={ps:.3f} ssim={ss:.3f}" if ps == ps else f"    {v:16s} (no finite metrics)"
        print(line, flush=True)
    print(f"[{args.tag}] output root: {save_dir}", flush=True)


if __name__ == "__main__":
    main()
