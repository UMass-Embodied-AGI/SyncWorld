# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.utils.data
import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io


def _json_num(v):
    """Coerce a metric value to a JSON-serializable number (fallback to str)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


@dataclass
class _LossRecord:
    loss: torch.Tensor | float = 0
    iter_count: int = 0
    name: str | None = None

    def reset(self) -> None:
        self.loss = 0
        self.iter_count = 0

    def get_stat(self, return_valid_mask_sum: bool = False) -> Tuple[float, float]:
        if self.iter_count == 0:
            self.loss = torch.tensor([float("nan")], device="cuda")
            self.iter_count = 1
        self.loss = self.loss.mean()
        msg_str = f"{self.name}: sum_loss={self.loss.item()}/iter_count={self.iter_count}="
        avg_loss_tensor = self.loss / self.iter_count
        # Create a mask (1 if valid, 0 if NaN or Inf)
        valid_mask = torch.tensor([torch.isfinite(avg_loss_tensor).float()], device="cuda")
        msg_str += f"avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}, "

        # Replace NaN/Inf with 0 to avoid affecting sum
        avg_loss_tensor = torch.where(
            torch.isfinite(avg_loss_tensor), avg_loss_tensor, torch.tensor([0.0], device="cuda")
        )

        # Reduce across all ranks
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)  # Sum of valid losses
        dist.all_reduce(valid_mask, op=dist.ReduceOp.SUM)  # Count of valid losses
        msg_str += f" | all_reduce: avg_loss={avg_loss_tensor.item()}, valid_mask={valid_mask.item()}"
        # Compute final average, avoiding division by zero
        if valid_mask.item() > 0:
            final_avg_loss = (avg_loss_tensor / valid_mask).item()
            valid_mask_sum = valid_mask.item()
        else:
            final_avg_loss = 0.0  # Default to zero if all values were invalid
            valid_mask_sum = 0

        avg_loss = final_avg_loss
        msg_str += f" | final: avg_loss={final_avg_loss}"
        if self.name is not None:
            log.debug(msg_str, rank0_only=False)
        self.reset()
        if return_valid_mask_sum:
            return avg_loss, valid_mask_sum
        else:
            return avg_loss


class WandbCallback(Callback):
    def __init__(
        self,
        logging_iter_multipler: int = 1,
        save_logging_iter_multipler: int = 1,
        save_s3: bool = False,
    ) -> None:
        super().__init__()
        self.final_loss_log = _LossRecord()
        self.final_loss_log_per_dataset = {}
        self.final_all_loss_log = {}
        self.logging_iter_multipler = logging_iter_multipler
        self.save_logging_iter_multipler = save_logging_iter_multipler
        assert self.logging_iter_multipler > 0, "logging_iter_multipler should be greater than 0"
        self.save_s3 = save_s3
        self.wandb_extra_tag = f"@{logging_iter_multipler}" if logging_iter_multipler > 1 else ""
        self.name = "wandb_loss_log" + self.wandb_extra_tag
        self.unstable_count = torch.zeros(1, device="cuda")

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if torch.isnan(loss) or torch.isinf(loss):
            log.critical(
                f"Unstable loss {loss} at iteration {iteration}",
                rank0_only=False,
            )
            self.unstable_count += 1

        dataset_name = data_batch.get("dataset_name", "default")

        # Handle case where dataset_name gets batched into a list
        if isinstance(dataset_name, list):
            # For reasoner, dataset_name will be a list of different datasets
            # For generator, dataset_name is a list of size larger than one,
            #   assume they are all the same
            # Dedup using set to extract identical dataset names
            dataset_name = list(set(dataset_name))

        if dataset_name == "default" and "__url__" in data_batch:
            # try to get the name from url
            dataset_name = ["/".join(data_batch["__url__"][0].split("/")[:-1])]

        for single_dataset_name in dataset_name:
            if single_dataset_name not in self.final_loss_log_per_dataset:
                self.final_loss_log_per_dataset[single_dataset_name] = _LossRecord()
                self.final_loss_log_per_dataset[single_dataset_name].name = single_dataset_name
            self.final_loss_log_per_dataset[single_dataset_name].loss += loss.detach().float()
            self.final_loss_log_per_dataset[single_dataset_name].iter_count += 1

        # VLM: per-sequence loss normalization using token counts when available
        if "avg_num_assistant_tokens" in output_batch:
            per_seq_loss = (
                loss
                * output_batch["avg_num_assistant_tokens"]
                * output_batch["batch_size_local"]
                / output_batch["current_num_assistant_tokens"]
            )
            per_seq_key = f"per_seq/{dataset_name}"
            if per_seq_key not in self.final_loss_log_per_dataset:
                self.final_loss_log_per_dataset[per_seq_key] = _LossRecord()
                self.final_loss_log_per_dataset[per_seq_key].name = per_seq_key
            self.final_loss_log_per_dataset[per_seq_key].loss += per_seq_loss
            self.final_loss_log_per_dataset[per_seq_key].iter_count += 1

        self.final_loss_log.loss += loss.detach().float()
        self.final_loss_log.iter_count += 1

        for key in output_batch.keys():
            # Curve can be plotted only on aggregated loss, not per-instance loss
            if "loss" in key and "per_instance" not in key:
                if key not in self.final_all_loss_log:
                    self.final_all_loss_log[key] = _LossRecord()
                self.final_all_loss_log[key].loss += output_batch[key].detach().float()
                self.final_all_loss_log[key].iter_count += 1

        if iteration % (self.config.trainer.logging_iter * self.logging_iter_multipler) == 0:
            avg_final_loss = self.final_loss_log.get_stat()

            # ``get_stat`` runs TWO dist.all_reduce collectives, so every rank must call it for the
            # SAME keys in the SAME order. ``final_all_loss_log`` is a per-rank dict keyed by FIRST
            # ENCOUNTER and is never reset, and the per-mode logging in ``omni_mot_model`` only emits
            # ``flow_matching_loss_{vision_fdm,action_idm}`` on steps where that rank's own batch
            # actually contains fdm / idm samples. With a small per-rank batch the ranks therefore
            # discover those keys at different steps, so the key ORDER (and briefly the key SET)
            # diverges -- and iterating raw ``.keys()`` pairs one rank's "sound" reduce with another
            # rank's "vision_fdm" reduce, scrambling the logged values (e.g. a nonzero
            # flow_matching_loss_sound on a sound_gen=False run). Training is unaffected: these
            # numbers are logging-only. Mirror the union+sort already applied to the per-dataset dict.
            local_loss_keys = list(self.final_all_loss_log.keys())
            gathered_loss_keys = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_loss_keys, local_loss_keys)
            union_loss_keys = set()
            for keys in gathered_loss_keys:
                union_loss_keys.update(keys)
            union_loss_keys = sorted(union_loss_keys)  # identical iteration order on every rank

            avg_final_all_loss = {}
            for key in union_loss_keys:
                record = self.final_all_loss_log.get(key)
                if record is None:
                    # This rank never saw the key. Use a THROWAWAY NaN record so it still joins the
                    # collective; get_stat masks NaN out of the cross-rank average. Deliberately NOT
                    # inserted into final_all_loss_log -- that dict is never reset, so a stored NaN
                    # would poison the key for the rest of the run.
                    record = _LossRecord()
                    record.loss = torch.tensor([float("nan")], device="cuda")
                    record.iter_count = 1
                avg_final_all_loss[key] = record.get_stat()

            # Step 1: Gather all dataset names across ranks
            local_dataset_names = list(self.final_loss_log_per_dataset.keys())
            all_dataset_names = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_dataset_names, local_dataset_names)

            # Step 2: Create the union of all dataset names
            union_dataset_names = set()
            for names in all_dataset_names:
                union_dataset_names.update(names)
            # Step 3: For any missing dataset name, add dummy _LossRecord with NaN loss
            union_dataset_names = sorted(list(union_dataset_names))  # This is very important!
            for dataset_name in union_dataset_names:
                if dataset_name not in self.final_loss_log_per_dataset:
                    dummy = _LossRecord()
                    dummy.loss += torch.tensor([float("nan")], device="cuda")  # Will be masked out
                    dummy.iter_count += 1
                    self.final_loss_log_per_dataset[dataset_name] = dummy

            avg_final_loss_per_dataset = {}
            for dataset_name in union_dataset_names:
                avg_loss, valid_mask_sum = self.final_loss_log_per_dataset[dataset_name].get_stat(
                    return_valid_mask_sum=True
                )
                if valid_mask_sum > 0:
                    avg_final_loss_per_dataset[dataset_name] = avg_loss

            dist.all_reduce(self.unstable_count, op=dist.ReduceOp.SUM)

            if distributed.is_rank0() and wandb.run is not None:
                info = {}
                info.update(
                    {
                        f"train{self.wandb_extra_tag}/loss": avg_final_loss,
                        f"train{self.wandb_extra_tag}/unstable_count": self.unstable_count.item(),
                        "iteration": iteration,
                    }
                )
                for key, loss in avg_final_all_loss.items():
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}_detail/{key}": loss,
                        }
                    )
                for dataset_name, loss in avg_final_loss_per_dataset.items():
                    tag = ""
                    if "per_seq" in dataset_name:
                        tag = "_per_seq"
                        dataset_name = dataset_name.replace("per_seq/", "")
                    info.update(
                        {
                            f"train{self.wandb_extra_tag}_per_data{tag}/{dataset_name}": loss,
                        }
                    )
                if self.save_s3:
                    if (
                        iteration
                        % (
                            self.config.trainer.logging_iter
                            * self.logging_iter_multipler
                            * self.save_logging_iter_multipler
                        )
                        == 0
                    ):
                        easy_io.dump(
                            info,
                            f"s3://rundir/{self.name}/Train_Iter{iteration:09d}.json",
                        )

                if wandb:
                    wandb.log(info, step=iteration)

                # Fallback: also persist the SAME metrics to a local JSONL under the checkpoint dir.
                # Keeps loss curves retrievable when the wandb server drops/loses history (append one
                # self-describing record per log step).
                _ckpt_dir = os.environ.get("CKPT_LOCAL_DIR", "").strip()
                if _ckpt_dir:
                    try:
                        os.makedirs(_ckpt_dir, exist_ok=True)
                        rec = {"step": int(iteration)}
                        rec.update({k: _json_num(v) for k, v in info.items()})
                        with open(os.path.join(_ckpt_dir, "metrics.jsonl"), "a") as _mf:
                            _mf.write(json.dumps(rec) + "\n")
                    except Exception as _e:  # never break training on best-effort logging
                        print(f"[wandb_log] metrics.jsonl append failed: {_e!r}", flush=True)

            # reset unstable count
            self.unstable_count.zero_()
            self.final_loss_log_per_dataset = {}
