# Differences from the paper

The experiments in the paper were run on a **Wan2.2** backbone. The released checkpoint is trained
on **Cosmos3-Nano** instead, so that the weights and the full training/evaluation stack could be
open-sourced under a single permissive license. Moving backbones changed a handful of things that
are worth knowing before you compare numbers or port a recipe.

Everything below describes the released recipe
([`examples/toml/sft_config/gripperhead_fdm_nano.toml`](../examples/toml/sft_config/gripperhead_fdm_nano.toml));
every value is a field of `GripperheadConfig` in
[`sft_config.py`](../cosmos_framework/configs/toml_config/sft_config.py).

## 1. Action units: metres → centimetres

The paper's implementation fed positions in **metres** and rotations in **degrees**. The released
model uses **centimetres and degrees**:

```python
action_translation_scale = 100.0     # m  -> cm
action_rotation_scale    = 57.2958   # rad -> deg
```

A typical per-step end-effector translation is on the order of `0.01 m`, while the paired rotation
deltas are single- or double-digit degrees. Mixing those two scales in one action vector leaves the
translation channels two orders of magnitude smaller than the rotation channels, so they contribute
almost nothing to the loss and are the first thing the model stops tracking. Rescaling translation
to centimetres puts both groups in a comparable numeric range.

This is a **units change, not a semantics change** — but it means an action vector built for the
paper's model is off by 100× in its first three components if fed to this one.

## 2. Action conditioning: cross-attention → joint self-attention

This is the substantive architectural difference.

In the paper's Wan2.2 implementation the action sequence is encoded into a per-latent-frame
embedding and handed to every transformer block **alongside** the hidden states, as a separate
conditioning signal consumed by cross-attention. Video tokens attend *to* actions; actions never
attend to anything.

Cosmos3 is a **Mixture-of-Transformers (MoT)**, where each modality is a first-class token stream.
Actions are projected into the shared token space (`action2llm` / `llm2action`), tagged with their
own modality and positional embeddings (`action_modality_embed`, `action_pos_embed`), and **packed
into the same sequence as the video and text tokens**. All modalities then interact through one
**joint self-attention** over the packed sequence, with a unified 3D-MRoPE aligning action tokens
to the video frames they belong to.

Two practical consequences:

- Conditioning is bidirectional. Action tokens see video context, which is what makes the same
  stack usable for **inverse dynamics** (video → actions) without a separate architecture — set
  `gripperhead.task_mode`.
- The action heads are the only newly-initialized parameters when starting from the Cosmos3-Nano
  base. That is why `gripperhead.resume_action_heads=false` matters on a fresh run and
  `true` when continuing from the released checkpoint.

## 3. Calibration: 12 segments → 6

The paper conditions on **12** calibration segments — both directions of each of the 6
end-effector DoFs. The released model uses **6**, one slot per DoF:

```python
calib_segments = 6      # x, y, z, yaw, pitch, roll
calib_seg_len  = 5      # frames per segment
```

Halving the calibration context roughly halves the tokens spent on it, which is a large share of
the sequence at 512×512. Segments are always emitted in the canonical order
`(x, y, z, yaw, pitch, roll)`, so the model sees each DoF in a fixed slot.

Each slot is required to be a **positive** sweep of its DoF, and positivity is judged on the
**body-frame action** — the quantity the model actually receives — rather than on the world-frame
pose series the runs are detected from. `gripperhead.calib_positive_body_actions` (on by default)
scores all twelve candidate runs (6 axes × both sweep directions) and gives slot *k* the one that
moves most positively along DoF *k*.

Scoring across all twelve rather than just the two runs of axis *k* matters once axis augmentation
is active: augmentation right-multiplies the rotation, so the body-frame action picks up `Sᵀ`,
which both flips signs and **permutes** axes. The demonstration that reads as `+x` after
augmentation may be the raw `-z` run, which a per-axis search would never consider. Detection stays
on the raw poses — that keeps monotone-run detection and the `move_range.pkl` signs valid — while
the selection is scored on the augmented poses.

The setting has to be the same at training and evaluation time; evaluation exposes it as
`--calib-positive-actions` / `--no-calib-positive-actions`. It has no effect at
`calib_segments = 12`, which emits both directions explicitly.

See [Visual Calibration](./visual_calibration.md) for how the sweeps are produced and segmented.
