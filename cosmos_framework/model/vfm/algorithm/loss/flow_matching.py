# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rectified-flow matching loss (vision / action / sound modalities).

Extracted from OmniMoTModel._compute_flow_matching_loss. The loss math is
unchanged; the only structural change is that ``tensor_kwargs_fp32`` is now
passed explicitly instead of being read from ``self``.
"""

from __future__ import annotations

import torch

from cosmos_framework.model.vfm.diffusion.rectified_flow import RectifiedFlow


def compute_flow_matching_loss(
    pred: list[torch.Tensor],
    target: list[torch.Tensor],
    condition_mask: list[torch.Tensor],
    timesteps: torch.Tensor,
    has_valid_tokens: bool,
    rectified_flow: RectifiedFlow,
    tensor_kwargs_fp32: dict,
    loss_scale: float | None = None,
    raw_action_dim: list[torch.Tensor] | None = None,
    normalize_by_active: bool = False,
    last_channel_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flow matching loss for a modality.

    Args:
        pred: Predicted velocity field (list of tensors, one per sample).
        target: Target velocity field (list of tensors, one per sample).
            Under rectified flow the target is ``v = eps - x0``.
        condition_mask: Mask where 1 = clean/conditioning, 0 = noisy/generation (list of tensors).
        timesteps: Diffusion timesteps for time weighting. Shape [B,1] for
            base/teacher_forcing (all frames share one timestep) or [B,T_max]
            for diffusion_forcing (per-frame independent timesteps). Time weights
            are applied per-frame before averaging, so non-uniform weight functions
            are handled correctly.
        has_valid_tokens: Whether this modality has valid noisy tokens.
        rectified_flow: The rectified flow object for time weighting.
        tensor_kwargs_fp32: Dict of dtype/device kwargs forwarded to
            ``rectified_flow.train_time_weight``.
        loss_scale: Optional per-modality loss scale. Falls back to the global
            ``rectified_flow_training_config.loss_scale`` when *None*.
            (Currently unused inside the function body — scaling is applied at the
            call site in ``OmniMoTModel._compute_losses``. Kept in the signature
            to preserve the original API.)
        normalize_by_active: When True, normalize per-instance loss by the count of
            active (noisy) elements rather than all elements. Preserves the
            ``sum / active_count`` semantics needed for distillation critics where
            conditioned frames contribute no signal and should not dilute the
            denominator.

    Returns:
        tuple: A tuple containing two elements:
            - Flow matching loss (or dummy loss for gradient consistency).
            - Per-instance loss (or dummy loss for gradient consistency).
    """
    if not has_valid_tokens:
        # Dummy loss to maintain backward graph consistency across ranks
        dummy_loss = 0.0 * sum(p.sum() for p in pred)
        return dummy_loss, dummy_loss.unsqueeze(0)  # make per-instance loss 1-D

    # condition_mask[i] is T-first with trailing singletons: [T,1,1] vision, [T,1] action.
    # tw_i gets the same shape so w(σ_t) broadcasts element-wise over non-T dims.
    per_instance_losses = []
    per_instance_weighted_losses = []
    active_counts = []          # per-item active (noised) token count; REAL (0 for conditioning-only items)
    weighted_active_sums = []   # per-item time-weighted generated-token error SUM (pre-normalization)

    for i in range(len(pred)):
        T_i = condition_mask[i].shape[0]
        sqerr_i = (pred[i] - target[i]) ** 2  # vision:[C,T,H,W]  action/sound:[T,D]
        noisy_mask_i = 1.0 - condition_mask[i]  # vision:[T,1,1]  action/sound:[T,1]
        if raw_action_dim is not None and raw_action_dim[i] is not None:
            sqerr_i = sqerr_i[:, : raw_action_dim[i]]
            if last_channel_weight != 1.0 and sqerr_i.shape[1] >= 1:
                # Up/down-weight the LAST action channel (the gripper) in the flow-matching loss.
                # The raw action layout is [Δpos, Δrot, gripper], so the last column is the gripper.
                cw = torch.ones(sqerr_i.shape[1], dtype=sqerr_i.dtype, device=sqerr_i.device)
                cw[-1] = last_channel_weight
                sqerr_i = sqerr_i * cw  # broadcasts over [T, D]
        if normalize_by_active:
            active_i = noisy_mask_i.sum() * (sqerr_i.numel() // noisy_mask_i.numel())  # real count (may be 0)
            active_counts.append(active_i)
            per_instance_losses.append((sqerr_i * noisy_mask_i).sum() / active_i.clamp(min=1))  # []
        else:
            per_instance_losses.append((sqerr_i * noisy_mask_i).mean())  # []

        ts_i = timesteps[i, :T_i] if timesteps.dim() > 1 else timesteps[i]  # DF:[T_i]  TF:[1]
        tw_i = rectified_flow.train_time_weight(ts_i, tensor_kwargs_fp32)  # DF:[T_i]  TF:[1]
        tw_i = tw_i.reshape(-1, *([1] * (condition_mask[i].ndim - 1)))  # vision:[T_i,1,1]  action/sound:[T_i,1]
        if normalize_by_active:
            wsum_i = (sqerr_i * tw_i * noisy_mask_i).sum()
            weighted_active_sums.append(wsum_i)
            per_instance_weighted_losses.append(wsum_i / active_i.clamp(min=1))
        else:
            per_instance_weighted_losses.append((sqerr_i * tw_i * noisy_mask_i).mean())

    per_instance_loss = torch.stack(per_instance_losses)  # [B]
    per_instance_weighted_loss = torch.stack(per_instance_weighted_losses)  # [B]
    if normalize_by_active:
        # GLOBAL active-token normalization across items: total (time-weighted) generated-token error
        # over ALL items / total number of generated tokens over ALL items. Fully-conditioning items
        # (calibration / history / source frames) have 0 active tokens, so they add 0 to BOTH the
        # numerator and denominator and cannot dilute the loss — unlike the plain per-item mean below,
        # which averages them in as zeros. For a single item this equals per_instance_weighted_loss.mean(),
        # so single-item callers (e.g. distillation critics) are unchanged.
        final_loss = torch.stack(weighted_active_sums).sum() / torch.stack(active_counts).sum().clamp(min=1)
    else:
        final_loss = per_instance_weighted_loss.mean()  # []
    return (
        final_loss,  # []
        per_instance_loss,  # [B]
    )
