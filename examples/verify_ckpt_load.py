#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Debug: does the inference model actually load the trained DCP weights, or does allow_partial_load
silently skip them (key/shape mismatch -> near-untrained model -> low PSNR)?

Compares the inference model's state_dict keys vs the checkpoint's .metadata keys, reports
matched/skipped/unused counts + trained-module coverage, and verifies a few trained tensors actually
change value after the load (vs their init)."""
from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import json
from pathlib import Path

import attrs
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

_REWRITES = [
    ("cosmos3._src.vfm.configs.base.", "cosmos_framework.configs.base."),
    ("cosmos3._src.vfm.models.", "cosmos_framework.model.vfm."),
    ("cosmos3._src.vfm.tokenizers.", "cosmos_framework.model.vfm.tokenizers."),
    ("cosmos3._src.imaginaire.", "cosmos_framework."),
]


def cfg_from(d: Path) -> Cosmos3OmniConfig:
    cfg = d / "config.json" if (d / "config.json").exists() else d / "model" / "config.json"
    text = cfg.read_text()
    for a, b in _REWRITES:
        text = text.replace(a, b)
    return Cosmos3OmniConfig(model=json.loads(text)["model"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config-dir", required=True)
    ap.add_argument("--weights", required=True, help="the iter_*/model DCP dir")
    args = ap.parse_args()

    config = cfg_from(Path(args.base_config_dir))
    config.parallelism = attrs.asdict(ParallelismConfig())
    config.compile = attrs.asdict(CompileConfig(enabled=False))
    model = Cosmos3OmniModel(config)
    sd = get_model_state_dict(model.model)
    model_keys = set(sd.keys())

    md = FileSystemReader(args.weights).read_metadata()
    ckpt_keys = set(md.state_dict_metadata.keys())

    matched = model_keys & ckpt_keys
    model_only = model_keys - ckpt_keys   # in model, NOT in ckpt -> stays at init (skipped)
    ckpt_only = ckpt_keys - model_keys    # in ckpt, NOT in model -> unused
    print(f"[verify] model keys={len(model_keys)}  ckpt keys={len(ckpt_keys)}")
    print(f"[verify] matched={len(matched)}  model_only(skipped->init)={len(model_only)}  ckpt_only(unused)={len(ckpt_only)}")
    for pat in ["moe_gen", "action2llm", "llm2action", "action_modality_embed", "vae2llm", "llm2vae",
                "time_embedder", "language_model", "visual"]:
        mk = {k for k in model_keys if pat in k}
        ck = {k for k in ckpt_keys if pat in k}
        print(f"    {pat:20s} model={len(mk):4d} ckpt={len(ck):4d} matched={len(mk & ck):4d}")
    print(f"[verify] sample model_only (skipped): {sorted(model_only)[:6]}")
    print(f"[verify] sample ckpt_only  (unused) : {sorted(ckpt_only)[:6]}")

    # shape mismatches among matched keys
    shape_mismatch = []
    for k in matched:
        try:
            if tuple(sd[k].shape) != tuple(md.state_dict_metadata[k].size):
                shape_mismatch.append((k, tuple(sd[k].shape), tuple(md.state_dict_metadata[k].size)))
        except Exception:
            pass
    print(f"[verify] shape mismatches among matched: {len(shape_mismatch)}" +
          (f"  e.g. {shape_mismatch[:3]}" if shape_mismatch else ""))

    # do a trained tensor's values actually change after load?
    probe = [k for k in matched if k.endswith("weight") and any(p in k for p in ("moe_gen", "action2llm", "vae2llm"))][:3]
    before = {k: float(sd[k].detach().float().norm()) for k in probe}
    dcp.load(state_dict=sd, storage_reader=FileSystemReader(args.weights),
             planner=DefaultLoadPlanner(allow_partial_load=True))
    for k in probe:
        aft = float(sd[k].detach().float().norm())
        print(f"[verify] {k}: norm init={before[k]:.4f} -> loaded={aft:.4f}  CHANGED={abs(aft-before[k])>1e-5}")


if __name__ == "__main__":
    main()
