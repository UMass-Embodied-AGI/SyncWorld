# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Map-style action SFT dataset: raw action dataset → ``ActionTransformPipeline``.

The wrapped dataset's ``__getitem__`` returns the raw sample
(``video``/``action``/``ai_caption``/``viewpoint``/``mode``/``domain_id``/
``idle_frames``). The model expects each sample to be passed through
``ActionTransformPipeline`` (spatial resize/pad, text tokenization, action
padding to ``max_action_dim``, and ``sequence_plan`` construction). This thin
wrapper composes the two so the experiment can hand a single map-style dataset
to ``RankPartitionedDataLoader`` (mirroring how the vision recipe uses
``get_sft_dataset``). The shipped producer is
:class:`~cosmos_framework.data.vfm.action.datasets.gripperhead_fdm_dataset.GripperheadFDMDataset`.
"""
from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline


class ActionSFTDataset(Dataset):
    """Wraps a map-style action dataset and applies ``ActionTransformPipeline`` per sample."""

    def __init__(self, dataset: Dataset, transform: ActionTransformPipeline, resolution: str | int | None):
        super().__init__()
        self._dataset = dataset
        self._transform = transform
        self._resolution = resolution

    # ``RankPartitionedDataLoader`` sets shard_rank / shard_world_size / shard_id on
    # the dataset it instantiates -- which is THIS wrapper. Forward them to the inner
    # dataset so rank-aware datasets (e.g. GripperheadFDMDataset) actually shard.
    @property
    def shard_rank(self):
        return getattr(self._dataset, "shard_rank", 0)

    @shard_rank.setter
    def shard_rank(self, value):
        setattr(self._dataset, "shard_rank", value)

    @property
    def shard_world_size(self):
        return getattr(self._dataset, "shard_world_size", 1)

    @shard_world_size.setter
    def shard_world_size(self, value):
        setattr(self._dataset, "shard_world_size", value)

    @property
    def shard_id(self):
        return getattr(self._dataset, "shard_id", 0)

    @shard_id.setter
    def shard_id(self, value):
        setattr(self._dataset, "shard_id", value)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._transform(self._dataset[idx], self._resolution)
