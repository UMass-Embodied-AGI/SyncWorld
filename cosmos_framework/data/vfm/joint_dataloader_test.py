# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.joint_dataloader import PackingDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L


class _FiniteActionDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        return {
            "sample_id": index,
            "text_token_ids": torch.tensor([1, 2]),
            "video": torch.zeros(3, 5, 32, 32),
            "action": torch.zeros(4, 7),
        }


def _sample_ids(batch: dict) -> list[int]:
    return [int(value.item()) for value in batch["sample_id"]]


def test_packing_dataloader_restarts_finite_source() -> None:
    source = L(torch.utils.data.DataLoader)(
        dataset=_FiniteActionDataset(4),
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    loader = PackingDataLoader(
        dataloader=source,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_samples_per_batch=4,
    )

    iterator = iter(loader)
    assert _sample_ids(next(iterator)) == [0, 1, 2, 3]
    assert _sample_ids(next(iterator)) == [0, 1, 2, 3]
