# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``gripperhead_fdm_nano`` — Cosmos3-Nano forward/inverse-dynamics pretraining on the
gripperhead folder format (RoboCasa / LIBERO / RLBench).

Forward dynamics: condition on ``num_history`` past frames + the full action (gripper-pose)
sequence, predict ``num_pred`` future frames. Inverse dynamics flips it: observe the video,
predict the action chunk. ``joint`` mixes the two per sample. All three are **instruction-free**
— every sample carries the task-neutral caption (see ``GripperheadFDMDataset.force_neutral_caption``).

**Where the numbers live.** Every recipe value in this file is read from ``GripperheadConfig``
(``cosmos_framework/configs/toml_config/sft_config.py``), which is the single source of truth for
the shipped recipe and is also the ``[gripperhead]`` TOML schema. Nothing here reads the
environment: the TOML (plus ``-- key=value`` CLI overrides) is the only way to change the recipe.
Three coupled values are derived rather than configured — ``axis_aug_enabled``,
``model.config.tokenizer.encode_exact_durations`` and ``checkpoint.keys_to_skip_loading`` — see
``sft_config.gripperhead_derived_overrides``.

Usage::

    GRIPPERHEAD_DATA_ROOTS=/path/to/robocasa_gripperhead_set/set_0 \\
    BASE_CHECKPOINT_PATH=<Cosmos3-Nano DCP dir> \\
    WAN_VAE_PATH=<Wan2.2_VAE.pth> \\
    torchrun --nproc_per_node=1 -m cosmos_framework.scripts.train \\
        --sft-toml=examples/toml/sft_config/gripperhead_fdm_nano.toml

See ``README.md`` ("Training") for the multi-GPU and multi-node commands.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.configs.toml_config.sft_config import (
    GripperheadConfig,
    gripperhead_derived_overrides,
)
from cosmos_framework.data.vfm.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.vfm.action.datasets.gripperhead_fdm_dataset import get_gripperhead_fdm_sft_dataset

cs = ConfigStore.instance()

# The shipped recipe. `load_experiment_from_toml` re-emits every one of these as a Hydra override
# from the (TOML + CLI) merged GripperheadConfig, so this instance is what you get when the
# experiment is composed directly without a TOML — identical either way.
R = GripperheadConfig()

# Fail fast at import time if the defaults are ever edited into an invalid combination (num_pred
# must be 4n, num_history must be 1 or 4n+1) rather than at the first VAE encode.
gripperhead_derived_overrides(R)

# Trainable parameter groups: the generation tower + the action heads. The action heads get 5x LR
# because they are the newest part of the network for a new embodiment.
_TRAINABLE_KEYS = [
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
]


gripperhead_fdm_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="gripperhead_fdm",
            name="gripperhead_fdm_nano",
            wandb_mode="disabled",
        ),
        model=dict(
            config=copy.deepcopy(NANO_MODEL_CONFIG),  # action_gen=True, max_action_dim=64
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=_TRAINABLE_KEYS,
            lr=5.0e-05,  # base LR; [optimizer].lr in the TOML overrides it
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="AdamW",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100000],  # match/exceed max_iter (TOML sets the real value)
            f_max=[1.0],
            f_min=[1.0],  # constant LR
            f_start=[1.0],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,  # smoke default (TOML overrides)
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Derived from [gripperhead].resume_action_heads; see gripperhead_derived_overrides.
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Nano DCP dir; supply via [checkpoint].load_path
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="gripperhead_fdm",
            max_samples_per_batch=4,  # count-based batch; [dataloader_train] in the TOML overrides
            max_sequence_length=None,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                # Shuffle the episode order (-> torch RandomSampler, reshuffled each epoch). WITHOUT
                # this the DataLoader uses a SequentialSampler over a domain-grouped (sorted /
                # concat'd rlbench->robocasa->robosuite) episode list, so each rank cycles through
                # domains/tasks in a fixed order -> a regular periodic loss oscillation.
                shuffle=True,
                datasets=dict(
                    gripperhead=dict(
                        ratio=1,
                        # Every kwarg below mirrors a [gripperhead] TOML field. Do NOT hardcode
                        # values here: edit GripperheadConfig so the schema, the dataset defaults
                        # and this node stay in lockstep.
                        dataset=L(get_gripperhead_fdm_sft_dataset)(
                            roots="???",  # comma-separated roots; supply via [gripperhead].roots
                            resolution=R.resolution,
                            episodes_cache_path=R.episodes_cache_path,
                            task_mode=R.task_mode,
                            joint_p_inverse=R.joint_p_inverse,
                            num_history_frames=R.num_history,
                            num_pred_frames=R.num_pred,
                            history_frame_stride=R.history_stride,
                            history_min_frames=R.history_min,
                            future_min_frames=R.future_min,
                            p_include_history=R.p_include_history,
                            fps=15.0,
                            use_wrist=R.use_wrist,
                            action_pose_convention=R.action_pose_convention,
                            action_rotation_format=R.action_rotation_format,
                            action_translation_scale=R.action_translation_scale,
                            action_rotation_scale=R.action_rotation_scale,
                            p_null_action=R.p_null_action,
                            use_calibration=R.use_calibration,
                            calib_num_segments=R.calib_segments,
                            calib_seg_len=R.calib_seg_len,
                            calib_frame_interval=R.calib_frame_interval,
                            p_include_calibration=R.p_include_calibration,
                            axis_aug_enabled=R.p_pose_axis_aug > 0.0,  # derived
                            axis_aug_mode=R.axis_aug_mode,
                            p_pose_axis_aug=R.p_pose_axis_aug,
                            photometric_aug_enabled=False,
                            contact_bias_prob=R.contact_bias_prob,
                            include_counterfactual=R.include_counterfactual,
                            include_perturb=R.include_perturb,
                            p_mask_history=R.p_mask_history,
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


_MODEL_CFG = gripperhead_fdm_nano["model"]["config"]

# Pin the VAE encode durations to the exact clip lengths this recipe produces (current+future,
# sparse history, calibration segment). Derived from the same three fields in
# gripperhead_derived_overrides, which re-emits this after the TOML + CLI merge.
_MODEL_CFG["tokenizer"]["encode_exact_durations"] = sorted(
    {R.num_pred + 1}
    | ({R.num_history} if R.num_history > 1 else set())
    | ({R.calib_seg_len} if R.use_calibration else set())
)

# Normalize the rectified-flow loss by the number of tokens actually being optimized (the noised /
# generated latents) rather than ALL vision tokens. Without this, adding conditioning items
# (history, calibration) inflates the denominator with zero-error conditioning tokens — deflating
# the reported loss ~N-fold and diluting the generation gradient. Scoped to gripperhead
# (model.config is a deepcopy; the global default stays False).
_MODEL_CFG["rectified_flow_training_config"]["normalize_loss_by_active"] = True

# Loss weights + teacher/student distillation, all from [gripperhead]. action_loss_weight overrides
# the NANO default (10.0), which was tuned for scaling UP a tiny action loss; gripperhead
# inverse/joint has a LARGE action loss to scale DOWN so it does not swamp the video loss.
_MODEL_CFG["rectified_flow_training_config"]["action_loss_weight"] = R.action_loss_weight
_MODEL_CFG["rectified_flow_training_config"]["gripper_loss_weight"] = R.gripper_loss_weight
_MODEL_CFG["rectified_flow_training_config"]["distill_lambda_cons"] = R.lambda_cons
_MODEL_CFG["rectified_flow_training_config"]["distill_p_cons"] = R.p_cons


for _item in [gripperhead_fdm_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
