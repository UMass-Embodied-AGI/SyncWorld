# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Auto-imported by every Python process started with PYTHONPATH=.
import os

# Opt-in (COSMOS_DL_FILE_SYSTEM_SHARING=1): switch torch's DataLoader IPC from the
# default 'file_descriptor' strategy (which stages worker tensors in /dev/shm) to
# 'file_system'. On shm-constrained containers, large video batches overflow the
# small /dev/shm tmpfs and a worker dies mid-transfer -> the main process then sees
# "unable to open shared memory object ... No such file or directory". 'file_system'
# sidesteps /dev/shm entirely. Guarded so non-training processes never import torch.
if os.environ.get("COSMOS_DL_FILE_SYSTEM_SHARING") == "1":
    try:
        import torch.multiprocessing as _tmp

        _tmp.set_sharing_strategy("file_system")
    except Exception:
        pass
