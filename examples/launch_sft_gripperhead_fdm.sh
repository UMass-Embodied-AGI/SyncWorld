#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Thin launcher for the gripperhead forward/inverse-dynamics pretraining recipe.
#
# It only assembles the torchrun command — the recipe itself lives entirely in
# examples/toml/sft_config/gripperhead_fdm_nano.toml (schema: GripperheadConfig). Pass any Hydra
# override as an argument to this script and it is forwarded after the `--` separator:
#
#   ./examples/launch_sft_gripperhead_fdm.sh gripperhead.resolution=256 trainer.max_iter=100
#
# See README.md ("Training") for the single-GPU / multi-GPU / multi-node commands this wraps.
#
# Required environment (paths only):
#   GRIPPERHEAD_DATA_ROOTS   comma-separated dataset roots
#   BASE_CHECKPOINT_PATH     Cosmos3-Nano DCP dir (see cosmos_framework.scripts.convert_model_to_dcp)
#   WAN_VAE_PATH             Wan2.2_VAE.pth
# Optional:
#   QWEN3VL_ASSETS           local text-tokenizer dir (set it to train fully offline)
#   NPROC_PER_NODE           GPUs on this node (default 1)
#   SHARD_DEGREE             FSDP shard degree (default = NPROC_PER_NODE)
#   OUTPUT_ROOT              run output root (default <repo>/outputs/train)
#   MASTER_PORT              torchrun rendezvous port (default 29500)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOML="examples/toml/sft_config/gripperhead_fdm_nano.toml"

: "${NPROC_PER_NODE:=1}"
: "${SHARD_DEGREE:=${NPROC_PER_NODE}}"
: "${OUTPUT_ROOT:=${REPO}/outputs/train}"
: "${MASTER_PORT:=29500}"

for var in GRIPPERHEAD_DATA_ROOTS BASE_CHECKPOINT_PATH WAN_VAE_PATH; do
    [[ -n "${!var:-}" ]] || { echo "ERROR: $var is not set (see the header of this script)" >&2; exit 1; }
done
[[ -d "$BASE_CHECKPOINT_PATH" ]] || { echo "ERROR: BASE_CHECKPOINT_PATH not found: $BASE_CHECKPOINT_PATH" >&2; exit 1; }
[[ -f "$WAN_VAE_PATH" ]]         || { echo "ERROR: WAN_VAE_PATH not found: $WAN_VAE_PATH" >&2; exit 1; }
if [[ -z "${QWEN3VL_ASSETS:-}" ]]; then
    # The TOML interpolates ${oc.env:QWEN3VL_ASSETS}; an empty value would resolve to an invalid
    # path, so fall back to the HuggingFace repo id (needs network on the first run).
    export QWEN3VL_ASSETS="Qwen/Qwen3-VL-8B-Instruct"
fi

echo ">>> recipe:   $TOML"
echo ">>> roots:    $GRIPPERHEAD_DATA_ROOTS"
echo ">>> gpus:     $NPROC_PER_NODE (fsdp shard degree $SHARD_DEGREE)"
echo ">>> overrides: ${*:-<none>}"

cd "$REPO"
IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=. TOKENIZERS_PARALLELISM=false \
    torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
    -m cosmos_framework.scripts.train \
    --sft-toml="$TOML" \
    -- "model.config.parallelism.data_parallel_shard_degree=${SHARD_DEGREE}" "$@"
