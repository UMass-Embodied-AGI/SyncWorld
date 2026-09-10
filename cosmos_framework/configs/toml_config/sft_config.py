# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""TOML-readable pydantic schema for SFT experiment knobs.

Sibling of ``toml_config_helper.py`` — this file holds the pydantic models
(the schema the TOML must conform to). All conversion logic (TOML →
override list, ``PATH_REMAPS``, etc.) lives in ``toml_config_helper.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import tomllib
from pydantic import BaseModel, ConfigDict, Field

from cosmos_framework.configs.toml_config.toml_config_helper import (
    TASK_TO_BASE_CONFIG,
    build_hydra_overrides,
)

# Common config for every model in this file:
# - ``extra="forbid"``        → unknown TOML keys raise ValidationError (typo guard).
# - ``protected_namespaces=()`` → silence the ``model_*`` field-name warning so
#                                 the ``model:`` field on SFTExperimentConfig is allowed.
_PYDANTIC_MODEL_CONFIG = ConfigDict(extra="forbid", protected_namespaces=())


# ---------------------------------------------------------------- job
class JobConfig(BaseModel):
    """Run identity + meta-fields that pick the Hydra config tree to load."""

    model_config = _PYDANTIC_MODEL_CONFIG

    task: str = Field(
        default="vfm",
        description=(
            "META — chooses which make_config() to call: "
            "'vfm' → cosmos_framework/configs/base/config.py (video foundation model), "
            "'vlm' → cosmos_framework/configs/base/vlm/config.py (vision-language model). "
            "Also picks the path-remap rules in toml_config_helper.PATH_REMAPS."
        ),
    )
    experiment: str = Field(
        default="",
        description=(
            "META — names the Hydra experiment LazyDict registered in "
            "ConfigStore under experiment/<name>. Resolved at load time via "
            "the 'experiment=<name>' Hydra-CLI override "
            "(e.g. 'vision_sft_nano')."
        ),
    )
    project: str = Field(
        default="",
        description=(
            "Wandb project (team-level bucket). Flows to config.job.project "
            "and is what shows up under 'Projects' in the wandb UI."
        ),
    )
    group: str = Field(
        default="",
        description=(
            "Wandb group — sub-label under <project> for clustering related "
            "runs (e.g. 'sft', 'action_bridge'). Flows to config.job.group."
        ),
    )
    name: str = Field(
        default="",
        description=(
            "Wandb run name. Flows to config.job.name and forms part of the "
            "output-dir path: $IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>. "
            "Leave empty (or use Hydra ${now:%Y-%m-%d}_${now:%H-%M-%S}) to "
            "get an auto-timestamped subdir."
        ),
    )
    wandb_mode: str = Field(
        default="disabled",
        description=(
            "Wandb upload mode: 'online' (real-time, needs WANDB_API_KEY), "
            "'offline' (log locally, sync later with `wandb sync`), or "
            "'disabled' (no wandb at all)."
        ),
    )


# ---------------------------------------------------------------- model
class EMAConfig(BaseModel):
    """Exponential Moving Average of the generation-pathway weights.

    Lands at ``model.config.ema.*`` on both VFM and VLM. When enabled the
    trainer keeps a second fp32 copy of the trainable params updated as
    ``ema_w = (1 - rate^k) · w_curr + rate^k · ema_w_prev``. EMA weights
    are used for inference; the live weights keep training.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    enabled: bool = Field(
        default=True,
        description=(
            "Turn EMA tracking on/off. Full fine-tunes typically enable it; "
            "LoRA recipes leave it off because the adapter weights are tiny."
        ),
    )
    rate: float = Field(
        default=0.1,
        description=(
            "Base EMA decay rate. Lower = slower decay = EMA tracks the live "
            "weights more tightly. Effective per-step rate is ramped by the "
            "iteration counter so the EMA 'warms up' from init."
        ),
    )
    iteration_shift: int = Field(
        default=0,
        description=(
            "Step offset added before computing the warmup ramp. Use a "
            "positive value when resuming so the EMA doesn't reset to "
            "'early-iter' decay strength."
        ),
    )


class ParallelismConfig(BaseModel):
    """FSDP / context-parallel / classifier-free-guidance topology.

    Lands at ``model.config.parallelism.*`` on both VFM and VLM.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    data_parallel_shard_degree: int = Field(
        default=-1,
        description=(
            "FSDP shard degree. -1 = auto-fit WORLD_SIZE from torchrun. "
            "Set explicitly when you want the run to fail loudly on the "
            "wrong GPU count."
        ),
    )
    data_parallel_replicate_degree: int = Field(
        default=1,
        description=(
            "FSDP replicate degree (HSDP). >1 adds an outer replicate loop "
            "so the same shard topology runs N times in parallel; usually "
            "only needed for very large clusters."
        ),
    )
    context_parallel_shard_degree: int = Field(
        default=1,
        description=(
            "Context-parallel shard degree. >1 splits the sequence dimension "
            "across this many ranks, which lets long-context models fit in "
            "memory. Used by super-tier configs (DP=4, CP=2 → 8 GPUs)."
        ),
    )
    cfg_parallel_shard_degree: int = Field(
        default=1,
        description=(
            "Classifier-free-guidance parallel shard degree. Splits the "
            "duplicated conditional/unconditional forward across ranks. "
            "Almost always 1 for SFT."
        ),
    )


class CompileConfig(BaseModel):
    """torch.compile knobs.

    Lands at ``model.config.compile.*`` on both VFM and VLM. These two
    fields used to live on ``ParallelismConfig`` as ``use_torch_compile``
    and ``compile_dynamic``; the rename is the only behavior change.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    enabled: bool = Field(
        default=False,
        description=(
            "torch.compile the network (was ``parallelism.use_torch_compile``). "
            "Big speedup on stable shapes; conflicts with some custom CUDA "
            "kernels and deterministic modes."
        ),
    )
    compile_dynamic: bool = Field(
        default=True,
        description=(
            "When enabled=True, recompile per-shape rather than specializing "
            "for one static shape. Required for the compile_tokenizer "
            "callback's progressive warmup."
        ),
    )


class ActivationCheckpointingConfig(BaseModel):
    """Recompute activations during backward to trade FLOPs for memory.

    Lands at ``model.config.activation_checkpointing.*`` on both VFM and VLM.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    mode: str = Field(
        default="full",
        description=(
            "AC mode: 'selective' (per-op SAC; save matmuls/FMHA, recompute "
            "the rest — MoT path only), 'full' (checkpoint each whole "
            "transformer block), or 'none' (no checkpointing — fastest but "
            "highest memory)."
        ),
    )
    save_ops_regex: list[str] = Field(
        default_factory=lambda: ["fmha"],
        description=(
            "Regex patterns for ops to KEEP saved when mode='selective'. "
            "Ignored in 'full'/'none' mode. Default keeps flash/multi-head-"
            "attention outputs."
        ),
    )
    preserve_rng_state: bool = Field(
        default=True,
        description=(
            "Stash and restore CUDA RNG across recompute boundaries. Required "
            "for deterministic results vs. non-checkpointed runs; small slowdown."
        ),
    )
    determinism_check: str = Field(
        default="default",
        description=(
            "Forwarded to torch.utils.checkpoint. 'default' disables the "
            "extra determinism check; 'match' cross-checks recomputed "
            "activations against the original (debug-only, very slow)."
        ),
    )


class ModelTokenizerConfig(BaseModel):
    """Video tokenizer (VAE) settings. VFM only — VLM skips this sub-tree."""

    model_config = _PYDANTIC_MODEL_CONFIG

    vae_path: str = Field(
        default="pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
        description=(
            "Path to Wan2.2_VAE.pth. SFT recipes typically pass this via "
            "env interpolation: vae_path = '${oc.env:WAN_VAE_PATH}'."
        ),
    )


# ---------------------------------------------------------------- model.text_tokenizer
class TextTokenizerConfig(BaseModel):
    """Text/VLM tokenizer settings. VFM only — routed to
    ``model.config.vlm_config.tokenizer``, NOT to the video tokenizer above."""

    model_config = _PYDANTIC_MODEL_CONFIG

    pretrained_model_name: Optional[str] = Field(
        default=None,
        description=(
            "HuggingFace repo id OR a local directory holding the text-tokenizer / processor "
            "assets. Set this to a local path for offline runs (no network at train time): "
            "pretrained_model_name = '${oc.env:QWEN3VL_ASSETS}'. None = keep the recipe default."
        ),
    )


class BackboneConfig(BaseModel):
    """Foundation backbone settings. VLM only — VFM keeps its backbone
    wiring inline in the experiment Python (vlm_config.model_instance)
    and skips this sub-tree.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    model_name: str = Field(
        default="???",
        description=(
            "HF repo ID or local snapshot path of the VLM backbone "
            "(e.g. 'Qwen/Qwen3-VL-8B-Instruct'). Drives AutoConfig + "
            "AutoModel selection (architecture). Remapped to "
            "'model.config.policy.backbone.model_name' on VLM; skipped on "
            "VFM. Default '???' is the OmegaConf MISSING sentinel — "
            "recognized by build_hydra_overrides and skipped, so the "
            "experiment Python's default takes effect when the TOML omits "
            "[model.backbone]."
        ),
    )
    safetensors_path: str = Field(
        default="???",
        description=(
            "Optional local path to a .safetensors file (or directory) used "
            "for weight loading. When set, overrides the auto-downloaded "
            "snapshot under model_name; the architecture is still driven by "
            "model_name. Useful for pointing at a converted/finetuned "
            "checkpoint while keeping the public HF model_name for tokenizer "
            "and architecture discovery. Remapped to "
            "'model.config.policy.backbone.safetensors_path' on VLM; "
            "skipped on VFM. Default '???' = MISSING sentinel (omitted from "
            "overrides; falls back to '' from VLMConfig, which means "
            "'use the auto-downloaded model_name snapshot')."
        ),
    )


class ModelConfig(BaseModel):
    """Top-level model knobs.

    Lands at ``model.config.*`` on both VFM and VLM. After the ParallelismConfig
    split, the training-infra surface (parallelism, compile, activation_
    checkpointing, precision) lives at the same depth on both tasks; per-task
    leaf skips are handled by ``PATH_REMAPS``.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    precision: str = Field(
        default="bfloat16",
        description=(
            "Compute dtype for the network forward/backward "
            "(``MixedPrecisionPolicy.param_dtype``). Was "
            "``parallelism.precision`` before the ParallelismConfig split; "
            "lands at ``model.config.precision`` now."
        ),
    )
    max_num_tokens_after_packing: int = Field(
        default=13312,
        description=(
            "Token-packing target: max number of tokens after sequence "
            "packing. -1 disables the cap. VFM-only — VLM uses "
            "data_setting.max_tokens and policy.qwen_max_video_token_length."
        ),
    )
    joint_attn_implementation: str = Field(
        default="two_way",
        description=(
            "VFM attention layout: 'two_way' (separate U/G blocks with "
            "cross-attention), 'three_way' (adds a sparsity-aware third "
            "block — NATTEN), or 'flex' (legacy). Used when "
            "[job].task='vfm'; skipped on VLM."
        ),
    )
    attn_implementation: str = Field(
        default="cosmos",
        description=(
            "VLM HF attention impl: 'cosmos' (cosmos NATTEN/Blackwell-FMHA "
            "wrapper), 'flash_attention_2' (HF flash-attn-2), 'sdpa' "
            "(torch SDPA), or 'eager' (pure-python fallback). Used when "
            "[job].task='vlm'; skipped on VFM."
        ),
    )
    lora_enabled: bool = Field(
        default=False,
        description=(
            "Inject LoRA adapters into the generation pathway BEFORE FSDP "
            "wraps the network. Pair with optimizer.keys_to_select=['lora_'] "
            "(train only adapters) and checkpoint.keys_to_skip_loading=["
            "..., 'lora_'] (don't load missing adapter tensors). Used by "
            "SUPER-tier configs (e.g. vision_sft_super); NANO-tier leaves "
            "it off. Skipped on VLM."
        ),
    )
    lora_rank: int = Field(
        default=16,
        description=(
            "LoRA rank `r`. Adapter shape is (rank × hidden_dim) per target "
            "module. Standard values are 4, 8, 16, 32."
        ),
    )
    lora_alpha: int = Field(
        default=32,
        description=(
            "LoRA scaling factor. Effective magnitude of the adapter update "
            "is alpha/rank; rank=16 alpha=32 gives a 2× scale."
        ),
    )
    lora_target_modules: str = Field(
        default="q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen",
        description=(
            "Comma-separated substrings of param names that get a LoRA "
            "adapter. Defaults target the four MoE-gen projection matrices."
        ),
    )

    ema: EMAConfig = Field(default_factory=EMAConfig)
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    activation_checkpointing: ActivationCheckpointingConfig = Field(
        default_factory=ActivationCheckpointingConfig
    )
    tokenizer: ModelTokenizerConfig = Field(default_factory=ModelTokenizerConfig)
    text_tokenizer: TextTokenizerConfig = Field(default_factory=TextTokenizerConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)


# ---------------------------------------------------------------- optimizer
class OptimizerConfig(BaseModel):
    """AdamW-family optimizer parameters. Same shape on VFM and VLM (eps skipped on VLM)."""

    model_config = _PYDANTIC_MODEL_CONFIG

    betas: list[float] = Field(
        default_factory=lambda: [0.9, 0.99],
        description=(
            "Adam β1, β2 — gradient and squared-gradient EMAs. Standard pair "
            "is (0.9, 0.999); SFT recipes commonly use (0.9, 0.99) or "
            "(0.9, 0.95) for tighter tracking of recent gradients."
        ),
    )
    eps: float = Field(
        default=1.0e-8,
        description=(
            "Adam numerical stability epsilon. 1e-8 is the PyTorch default; "
            "1e-6 is sometimes used in bf16 to avoid underflow in the "
            "squared-gradient denominator. Skipped on VLM (no eps field)."
        ),
    )
    fused: bool = Field(
        default=True,
        description=(
            "Use the fused AdamW kernel. Faster on modern GPUs; slightly "
            "different numerical behavior vs. the foreach implementation."
        ),
    )
    keys_to_select: list[str] = Field(
        default_factory=list,
        description=(
            "Substring allowlist for params that the optimizer trains. "
            "Empty list = train everything. ['lora_'] = LoRA-only fine-tune "
            "(freezes everything except adapters)."
        ),
    )
    lr: float = Field(
        default=2.0e-4,
        description="Base learning rate.",
    )
    lr_multipliers: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-param-group LR multipliers (substring → multiplier). Used "
            "by action recipes to e.g. give 'action_modality_embed' 5× the "
            "base lr. Substrings not in the dict default to 1.0."
        ),
    )
    weight_decay: float = Field(
        default=0.0,
        description="AdamW decoupled weight decay. 0 disables.",
    )


# ---------------------------------------------------------------- scheduler
class SchedulerConfig(BaseModel):
    """LambdaLinear / LambdaCosine LR scheduler knobs.

    All four ``f_*`` values are **ratios of the optimizer's base lr** —
    effective lr at the corresponding milestone = ``lr × f_x``. Each list
    has one entry per scheduler cycle.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    cycle_lengths: list[int] = Field(
        default_factory=lambda: [20000],
        description=(
            "Length of each cycle in optimizer steps. With one entry, the "
            "scheduler completes one full warmup→peak→trough cycle over "
            "that many iterations."
        ),
    )
    f_max: list[float] = Field(
        default_factory=lambda: [1.0],
        description="Peak LR multiplier reached at the end of warmup.",
    )
    f_min: list[float] = Field(
        default_factory=lambda: [0.0],
        description=(
            "Final LR multiplier at the end of each cycle (the 'floor'). "
            "For LambdaCosine the LR decays toward lr × f_min."
        ),
    )
    f_start: list[float] = Field(
        default_factory=lambda: [1.0e-6],
        description=(
            "Initial LR multiplier at step 0, before warmup ramps up."
        ),
    )
    verbosity_interval: int = Field(
        default=0,
        description=(
            "How often the scheduler logs the current LR (in optimizer "
            "steps). 0 = silent. VFM only — skipped on VLM."
        ),
    )
    warm_up_steps: list[int] = Field(
        default_factory=lambda: [100],
        description=(
            "Linear warmup duration in optimizer steps. LR ramps from "
            "lr × f_start to lr × f_max linearly over this many iters "
            "before the cosine/linear decay begins."
        ),
    )


# ---------------------------------------------------------------- trainer
class CompileTokenizerCallback(BaseModel):
    """Lazy ``torch.compile`` of the VAE tokenizer once shapes stabilize.

    VFM only — skipped on VLM (no tokenizer to compile).
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    compile_after_iterations: int = Field(
        default=3,
        description=(
            "Wait this many training iterations after start before triggering "
            "the compile (lets one-shot init / dataloader settle)."
        ),
    )
    enabled: bool = Field(
        default=True,
        description="Master switch for the callback.",
    )
    warmup_resolutions: Optional[list[str]] = Field(
        default=None,
        description=(
            "Resolutions to 'prime' the compile cache with. The callback "
            "runs the tokenizer once per listed resolution so the compiled "
            "graph for each is ready before training hits it. None = use "
            "whatever resolutions the tokenizer's encode_chunk_frames knows."
        ),
    )


class GradClipCallback(BaseModel):
    """Gradient clipping callback. Present on both VFM and VLM."""

    model_config = _PYDANTIC_MODEL_CONFIG

    clip_norm: float = Field(
        default=1.0,
        description=(
            "Maximum global L2 norm of the gradient. Steps with a larger "
            "norm are rescaled so ||grad|| ≤ clip_norm."
        ),
    )
    force_finite: bool = Field(
        default=True,
        description=(
            "When True, replace NaN/Inf grads with zero before the step "
            "(treats them as no-op rather than crashing). VFM defaults to "
            "True; VLM defaults to False."
        ),
    )


class TrainerCallbacksConfig(BaseModel):
    """Only the two callbacks the schema currently surfaces. The full
    callbacks dict (norm_monitor, mfu, heart_beat, …) stays in the
    experiment Python.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    compile_tokenizer: CompileTokenizerCallback = Field(default_factory=CompileTokenizerCallback)
    grad_clip: GradClipCallback = Field(default_factory=GradClipCallback)


class TrainerConfig(BaseModel):
    """Trainer-level knobs that the TOML drives directly."""

    model_config = _PYDANTIC_MODEL_CONFIG

    distributed_parallelism: str = Field(
        default="fsdp",
        description=(
            "Distributed strategy. 'fsdp' (the only supported value today) "
            "routes through cosmos's FSDP wrapper."
        ),
    )
    grad_accum_iter: int = Field(
        default=1,
        description=(
            "Number of micro-batches accumulated before each "
            "optimizer.step(). Effective global batch = grad_accum_iter × "
            "per-rank batch × world_size."
        ),
    )
    logging_iter: int = Field(
        default=50,
        description="Console / wandb log frequency (in optimizer steps).",
    )
    max_iter: int = Field(
        default=500,
        description="Total number of optimizer steps the run will execute.",
    )
    callbacks: TrainerCallbacksConfig = Field(default_factory=TrainerCallbacksConfig)


# ---------------------------------------------------------------- checkpoint
class CheckpointConfig(BaseModel):
    """Resume + save policy. Lands at ``config.checkpoint.*``."""

    model_config = _PYDANTIC_MODEL_CONFIG

    keys_to_skip_loading: list[str] = Field(
        default_factory=list,
        description=(
            "Substring blocklist applied at load time. Any tensor whose FQN "
            "contains one of these substrings is skipped (kept at fresh-"
            "init). Used to mask EMA + LoRA + action layers when "
            "warm-starting from a base checkpoint without them."
        ),
    )
    load_path: str = Field(
        default="???",
        description=(
            "Path to the checkpoint directory to load. '???' is the "
            "OmegaConf MISSING sentinel — recognized by build_hydra_overrides "
            "and skipped, so the user must provide a real path at runtime "
            "(via env interpolation or CLI extra-override)."
        ),
    )
    save_iter: int = Field(
        default=100,
        description="Save a new checkpoint every N optimizer steps.",
    )


# ---------------------------------------------------------------- dataloader_train
class DataloaderTrainConfig(BaseModel):
    """Top-level dataloader scalars only. The dataloader's class (LazyCall)
    and full pipeline wiring (datasets, packers, …) stay in the experiment
    Python — they vary too much between VFM IterativeJointDataLoader,
    PackingDataLoader, and VLM DataPackerDataLoader to model uniformly.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    max_samples_per_batch: Optional[int] = Field(
        default=None,
        description=(
            "Cap on samples per micro-batch. Remapped to 'max_batch_size' "
            "on the VLM DataPackerDataLoader. None = no per-count cap "
            "(the packer's token budget is what limits batch size)."
        ),
    )
    max_sequence_length: Optional[int] = Field(
        default=None,
        description=(
            "Cap on tokens per packed sequence. Remapped to 'max_tokens' "
            "on the VLM DataPackerDataLoader. None = no per-token cap."
        ),
    )
    max_caption_tokens: Optional[int] = Field(
        default=None,
        description=(
            "VFM only. Per-caption token cap before truncation — remapped to the SFT "
            "dataset's 'max_caption_tokens'. Structured-JSON captions are longer than dense "
            "prose, so the example recipes set 2048 (measured max ~1790). None = keep the "
            "recipe default. Skipped on VLM (the data packer caps via max_sequence_length)."
        ),
    )
    seed: int = Field(
        default=42,
        description=(
            "Dataloader RNG seed. Skipped on VLM (DataPackerDataLoader has "
            "no seed ctor kwarg there)."
        ),
    )


# ---------------------------------------------------------------- gripperhead
class GripperheadConfig(BaseModel):
    """The gripperhead forward/inverse-dynamics (FDM/IDM) pretraining recipe.

    **This class is the single source of truth for the shipped recipe.** Every default below is
    the value used by the reference pretraining run; ``GripperheadFDMDataset`` /
    ``get_gripperhead_fdm_sft_dataset`` mirror them and
    ``gripperhead_recipe_defaults_test.py`` fails if the two ever drift apart.

    Most fields are routed straight onto the dataset node
    (``dataloader_train.dataloader.datasets.gripperhead.dataset.*``) by ``PATH_REMAPS``; the
    handful that feed the model / checkpoint instead are remapped individually, and four values
    are *derived* rather than configured — see ``gripperhead_derived_overrides``:

    - ``dataset.axis_aug_enabled``            <- ``p_pose_axis_aug > 0``
    - ``model.config.tokenizer.encode_exact_durations``
                                              <- ``num_pred``, ``num_history``, ``calib_seg_len``
    - ``checkpoint.keys_to_skip_loading``     <- ``resume_action_heads``

    Override any field from the CLI without touching the TOML, e.g.::

        torchrun ... --sft-toml=<toml> -- gripperhead.resolution=256 gripperhead.num_history=9
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    # --- data source -------------------------------------------------------------------
    roots: str = Field(
        default="",
        description=(
            "Comma-separated gripperhead dataset roots. Recipes pass this via env "
            "interpolation: roots = '${oc.env:GRIPPERHEAD_DATA_ROOTS}'. Each root is walked for "
            "``<task>/episode_*/<camera_poses_*>/{expert,counterfactual_*}`` leaves."
        ),
    )
    resolution: int = Field(
        default=512,
        description=(
            "Base square resolution tier; must exist in VIDEO_RES_SIZE_INFO. use_wrist switches "
            "to the wide '<res>_w2' tier (<res>h x 2*<res>w) automatically."
        ),
    )
    episodes_cache_path: Optional[str] = Field(
        default=None,
        description="Optional JSON episode-discovery cache; None = walk the roots every start-up.",
    )

    # --- task -------------------------------------------------------------------------
    task_mode: str = Field(
        default="forward_dynamics",
        description=(
            "forward_dynamics = past video + future actions -> generate future video. "
            "inverse_dynamics = all video -> predict the action chunk. "
            "joint = per-sample coin-flip between the two (P(inverse) = joint_p_inverse). "
            "All three are instruction-free: every sample gets the task-neutral caption."
        ),
    )
    joint_p_inverse: float = Field(
        default=0.5, description="task_mode='joint' only: P(a sample is inverse_dynamics)."
    )

    # --- horizon ----------------------------------------------------------------------
    num_history: int = Field(
        default=25,
        description=(
            "Sparse history frames. 1 = no separate history item (plain current+future clip); "
            ">1 = history is its own VAE item and must be 4n+1."
        ),
    )
    num_pred: int = Field(
        default=16,
        description=(
            "Dense future frames to generate. Must be a positive multiple of 4 so the "
            "current+future item (num_pred+1) is a valid VAE 4n+1 length."
        ),
    )
    history_stride: int = Field(
        default=3, description="Real-frame stride between sparse history frames."
    )
    history_min: int = Field(
        default=9,
        description="Minimum REAL history frames when calibration is null/dropped/off (0 when present).",
    )
    future_min: int = Field(
        default=5,
        description="Minimum REAL future frames sampled during training; the tail is last-frame padded.",
    )
    p_include_history: float = Field(
        default=0.8, description="P(include real history) on samples where calibration IS present."
    )

    # --- views ------------------------------------------------------------------------
    use_wrist: bool = Field(
        default=False,
        description="Stitch [main | wrist] side-by-side (2:1) and use the wide '<res>_w2' tier.",
    )

    # --- action representation --------------------------------------------------------
    action_pose_convention: str = Field(
        default="backward_framewise",
        description=(
            "backward_framewise = T_i^-1 T_{i+1} (per-frame deltas). "
            "backward_anchored = T_0^-1 T_{i+1} (everything relative to the current frame)."
        ),
    )
    action_rotation_format: str = Field(
        default="euler_xyz",
        description="rot6d | euler_xyz | quat_xyzw | axisangle | rot9d. Sets the action dim.",
    )
    action_translation_scale: float = Field(
        default=100.0, description="Translation unit scale; 100 = metres -> centimetres."
    )
    action_rotation_scale: float = Field(
        default=57.2958, description="Rotation unit scale; 57.2958 = radians -> degrees."
    )
    p_null_action: float = Field(
        default=0.1,
        description=(
            "P(null the future-driving action) for action-CFG; training only, and forced to 0 in "
            "pure inverse_dynamics (where the action IS the target). 1.0 = action-unconditional."
        ),
    )

    # --- calibration ------------------------------------------------------------------
    use_calibration: bool = Field(
        default=True,
        description=(
            "Prepend K calibration segments as separate conditioning items. Requires a sibling "
            "'calibration/' dir per episode."
        ),
    )
    calib_segments: int = Field(default=6, description="K: 6 (one per DoF) or 12 (both signs).")
    calib_seg_len: int = Field(default=5, description="Frames per calibration segment (1+4n or 4n).")
    calib_frame_interval: int = Field(
        default=3, description="Subsample stride on the raw calib clip before segment detection."
    )
    calib_positive_body_actions: bool = Field(
        default=True,
        description=(
            "Select each 6-segment calibration slot on the BODY-FRAME action, scoring all twelve "
            "candidate runs (6 axes x both sweep directions) so every slot is a positive sweep of "
            "its DoF; when axis augmentation is active the scoring uses the augmented poses. "
            "This flag MUST be the same at training and evaluation time. No effect when "
            "calib_segments=12, which emits both directions."
        ),
    )
    p_include_calibration: float = Field(
        default=0.9,
        description="P(include real calibration); otherwise a black null-placeholder + drop flag.",
    )

    # --- augmentation -----------------------------------------------------------------
    p_pose_axis_aug: float = Field(
        default=0.6,
        description=(
            "P(apply an axis/pose-basis relabel), one transform per sample shared by "
            "calib/history/current+future. 0 disables axis aug entirely (derives axis_aug_enabled)."
        ),
    )
    axis_aug_mode: str = Field(
        default="all", description="all | swap_only | flip_only | scale_only | none."
    )
    contact_bias_prob: float = Field(
        default=0.5,
        description="P(shift the window onto a robot-object contact onset/release from contacts.json).",
    )
    include_counterfactual: bool = Field(
        default=True, description="Also train on counterfactual_* (non-expert) clips per episode."
    )
    include_perturb: bool = Field(
        default=False, description="Also train on perturb_* clips (RLBench's non-expert naming)."
    )

    # --- masked-history distillation --------------------------------------------------
    p_mask_history: float = Field(
        default=0.3,
        description=(
            "P(move a recent contiguous run of the TARGET modality's real history tokens out of "
            "conditioning so they are predicted): forward -> history VIDEO, inverse -> history "
            "ACTION. Forces cross-modal inference instead of copying. 0 = off."
        ),
    )
    lambda_cons: float = Field(
        default=0.25,
        description=(
            "Weight of the teacher/student consistency loss. >0 rewrites the batch into "
            "consecutive (teacher, student=calibration-nulled) pairs sharing eps+timestep and adds "
            "a stop-grad consistency term on the teacher's generation region. Needs calibration "
            "items and an even per-rank batch. 0 = off."
        ),
    )
    p_cons: float = Field(
        default=0.5, description="Fraction of steps on which the consistency pairing is applied."
    )

    # --- loss weights -----------------------------------------------------------------
    action_loss_weight: float = Field(
        default=0.05,
        description=(
            "Multiplies the action flow-matching loss. The inverse-dynamics action loss lives in a "
            "much larger target space than the video loss, so joint training needs it scaled down. "
            "No-op in pure forward_dynamics."
        ),
    )
    gripper_loss_weight: float = Field(
        default=1.0,
        description="Extra multiplier on the GRIPPER channel (last action dim) of the action loss.",
    )

    # --- checkpoint behaviour ---------------------------------------------------------
    resume_action_heads: bool = Field(
        default=True,
        description=(
            "True = INHERIT the trained action heads from the checkpoint at checkpoint.load_path "
            "(continuing / forking a gripperhead run). False = re-initialize them fresh, which is "
            "what you want when load_path is the public Cosmos3-Nano base and the gripperhead "
            "embodiment is new. Derives checkpoint.keys_to_skip_loading."
        ),
    )


_ACTION_HEAD_KEYS = ["action2llm", "llm2action", "action_modality_embed", "action_pos_embed"]


def gripperhead_derived_overrides(cfg: GripperheadConfig) -> list[str]:
    """Hydra overrides for the values *derived* from ``[gripperhead]`` rather than set directly.

    Kept here (next to the schema) so there is exactly one place that knows how the recipe's
    scalars turn into the coupled structural values, instead of recomputing them at experiment
    import time from the environment.
    """
    if cfg.num_pred < 4 or cfg.num_pred % 4 != 0:
        raise ValueError(
            f"[gripperhead].num_pred must be a positive multiple of 4 so the current+future item "
            f"(num_pred+1) is a valid VAE 4n+1 length; got {cfg.num_pred}"
        )
    if cfg.num_history > 1 and (cfg.num_history - 1) % 4 != 0:
        raise ValueError(
            f"[gripperhead].num_history must be 1 or 4n+1 (the history is its own VAE item and the "
            f"Cosmos VAE temporal compression is 4); got {cfg.num_history}"
        )

    # The VAE must be told every exact clip length it will be asked to encode, or it pads and the
    # latent count stops matching the packer's plan.
    durations = {cfg.num_pred + 1}                 # current+future item (also the joint clip when H=1)
    if cfg.num_history > 1:
        durations.add(cfg.num_history)             # separate sparse-history item
    if cfg.use_calibration:
        durations.add(cfg.calib_seg_len)           # each calibration segment item

    dataset = "dataloader_train.dataloader.datasets.gripperhead.dataset"
    # net_ema is ALWAYS skipped (EMA warm-starts by copying net -> net_ema in dcp.py).
    skip = ["net_ema."] + ([] if cfg.resume_action_heads else _ACTION_HEAD_KEYS)
    return [
        f"{dataset}.axis_aug_enabled={'true' if cfg.p_pose_axis_aug > 0 else 'false'}",
        f"model.config.tokenizer.encode_exact_durations=[{','.join(str(d) for d in sorted(durations))}]",
        f"checkpoint.keys_to_skip_loading=[{','.join(skip)}]",
    ]


# ---------------------------------------------------------------- top
class SFTExperimentConfig(BaseModel):
    """Top-level structured-TOML schema. Each field corresponds to a
    top-level ``[<section>]`` block in the TOML and to a sub-model above.
    """

    model_config = _PYDANTIC_MODEL_CONFIG

    job: JobConfig = Field(default_factory=JobConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    dataloader_train: DataloaderTrainConfig = Field(default_factory=DataloaderTrainConfig)
    gripperhead: GripperheadConfig = Field(default_factory=GripperheadConfig)


# ---------------------------------------------------------------------------
# End-to-end loader: TOML → validate → Hydra overrides → merged Config.
# ---------------------------------------------------------------------------
def load_experiment_from_toml(
    toml_path: str | Path,
    extra_overrides: list[str] | None = None,
) -> Any:
    """End-to-end loader for the SFT structured-TOML schema.

    The base config module is picked from ``[job].task`` in the TOML:

    - ``task = "vfm"`` → ``cosmos_framework/configs/base/config.py``
    - ``task = "vlm"`` → ``cosmos_framework/configs/base/vlm/config.py``

    ``extra_overrides`` is appended after the TOML-derived Hydra overrides, so
    command-line entries take precedence over TOML values. Each entry must be
    Hydra dotted-path syntax (``key.path=value``); the ``--`` separator token
    is filtered out. Examples::

        ["optimizer.lr=1e-5", "trainer.max_iter=200"]
        ["model.config.parallelism.data_parallel_shard_degree=4"]

    Calls ``cosmos_framework.utils.config.load_config`` which:

    1. Imports the base config module and runs ``make_config()``. This
       registers every config group (model, ema, tokenizer, ...) and imports
       all experiment modules so their ``cs.store(group="experiment", ...)``
       side-effects fire.
    2. Runs ``override(config, overrides)`` — Hydra ``compose`` then resolves
       the ``experiment=<name>`` selector against ``ConfigStore`` and applies
       the dotted-path overrides we generated from the TOML, followed by
       ``extra_overrides``.

    Returns the merged ``Config`` instance, ready for ``launch()``.
    """
    with open(toml_path, "rb") as fh:
        raw = tomllib.load(fh)

    # Validate structure against the pydantic schema (raises ValidationError on
    # unknown keys because of ``extra="forbid"``).
    validated = SFTExperimentConfig.model_validate(raw)

    task = raw.get("job", {}).get("task", "vfm")
    try:
        base_config_path = TASK_TO_BASE_CONFIG[task]
    except KeyError as e:
        raise ValueError(
            f"{toml_path}: [job].task={task!r} is not supported. "
            f"Valid values: {sorted(TASK_TO_BASE_CONFIG)}"
        ) from e

    # Split the CLI overrides: ``gripperhead.*`` entries are folded back into the pydantic recipe
    # (below) instead of being passed through verbatim, so they get the same PATH_REMAPS routing as
    # the TOML block AND participate in the derived values. Everything else passes through as-is.
    cli_gripperhead: dict[str, str] = {}
    cli_other: list[str] = []
    for o in extra_overrides or []:
        # Filter "--" separator tokens (argparse may include them) and skip empty entries.
        # Hydra requires "key=value" shape — reject anything malformed early.
        if not o or o == "--":
            continue
        if "=" not in o:
            raise ValueError(
                f"extra override {o!r} must be Hydra dotted-path syntax (e.g. 'optimizer.lr=1e-5')."
            )
        key, _, value = o.partition("=")
        if key.strip().startswith("gripperhead."):
            field = key.strip()[len("gripperhead."):]
            if field not in GripperheadConfig.model_fields:
                raise ValueError(
                    f"unknown override {o!r}: [gripperhead] has no field {field!r}. "
                    f"Valid fields: {sorted(GripperheadConfig.model_fields)}"
                )
            cli_gripperhead[field] = value
        else:
            cli_other.append(o)

    # ``[gripperhead]`` is the one section whose defaults ARE the recipe, so emit the fully
    # populated (validated) block rather than only the keys the TOML happened to spell out. That
    # way the schema is the single source of truth and a partial TOML still produces the shipped
    # recipe, instead of silently falling back to whatever the experiment module happens to hold.
    gripperhead: Optional[GripperheadConfig] = None
    raw = dict(raw)
    if "gripperhead" in raw or raw.get("job", {}).get("experiment", "").startswith("gripperhead"):
        gripperhead = validated.gripperhead
        if cli_gripperhead:
            gripperhead = GripperheadConfig.model_validate(
                {**gripperhead.model_dump(), **cli_gripperhead}
            )
        # ``roots`` / ``episodes_cache_path`` are environment-specific paths, not recipe values;
        # drop them when empty so the experiment default (or an env interpolation) stands.
        raw["gripperhead"] = {
            k: v for k, v in gripperhead.model_dump().items() if v is not None and v != ""
        }
    elif cli_gripperhead:
        raise ValueError(
            "gripperhead.* overrides were given but the TOML has no [gripperhead] section and "
            f"[job].experiment does not name a gripperhead recipe: {sorted(cli_gripperhead)}"
        )
    else:
        raw.pop("gripperhead", None)

    overrides = build_hydra_overrides(raw)
    overrides += cli_other
    if gripperhead is not None:
        # LAST, so the derived values always reflect the final recipe (TOML + CLI).
        overrides += gripperhead_derived_overrides(gripperhead)

    # Import lazily so this module stays cheap to import in non-training contexts.
    from cosmos_framework.utils.config import load_config

    return load_config(base_config_path, overrides)
