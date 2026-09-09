# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import re
from typing import List, Tuple

IMAGE_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    # Our desired 256 resolution is the one below (commented).

    # Desired: "256": {"1,1": (336, 336), "4,3": (384, 288), "3,4": (288, 384), "16,9": (448, 256), "9,16": (256, 448)},
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {"1,1": (640, 640), "4,3": (736, 544), "3,4": (544, 736), "16,9": (832, 480), "9,16": (480, 832)},
    # 704 resolutions are nicely divisible by 32
    "704": {"1,1": (960, 960), "4,3": (1088, 832), "3,4": (832, 1088), "16,9": (1280, 704), "9,16": (704, 1280)},
    "720": {"1,1": (960, 960), "4,3": (1104, 832), "3,4": (832, 1104), "16,9": (1280, 720), "9,16": (720, 1280)},
    # 768 for arena.ai
    "768": {"1,1": (1024, 1024), "4,3": (1184, 880), "3,4": (880, 1184), "16,9": (1360, 768), "9,16": (768, 1360)},
    "1080": {"1,1": (1440, 1440), "4,3": (1664, 1248), "3,4": (1248, 1664), "16,9": (1920, 1080), "9,16": (1080, 1920)},
    "1280": {"1,1": (1712, 1712), "4,3": (1968, 1472), "3,4": (1472, 1968), "16,9": (2272, 1280), "9,16": (1280, 2272)},
    "2048": {
        "1,1": (2728, 2728),
        "4,3": (3160, 2368),
        "3,4": (2368, 3160),
        "16,9": (3640, 2048),
        "9,16": (2048, 3640),
    },
    "gt_2048": {
        "1,1": (5464, 5464),
        "4,3": (6304, 4728),
        "3,4": (4728, 6304),
        "16,9": (7280, 4096),
        "9,16": (4096, 7280),
    },
}

VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    # Our desired 256 resolution is the one below (commented).

    # Desired: "256": {"1,1": (336, 336), "4,3": (384, 288), "3,4": (288, 384), "16,9": (448, 256), "9,16": (256, 448)},
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {"1,1": (640, 640), "4,3": (736, 544), "3,4": (544, 736), "16,9": (832, 480), "9,16": (480, 832)},
    # 704 resolutions are nicely divisible by 32
    "704": {"1,1": (960, 960), "4,3": (1088, 832), "3,4": (832, 1088), "16,9": (1280, 704), "9,16": (704, 1280)},
    "720": {"1,1": (960, 960), "4,3": (1104, 832), "3,4": (832, 1104), "16,9": (1280, 720), "9,16": (720, 1280)},
    # 768 for arena.ai
    "768": {"1,1": (1024, 1024), "4,3": (1184, 880), "3,4": (880, 1184), "16,9": (1360, 768), "9,16": (768, 1360)},
    "1080": {"1,1": (1440, 1440), "4,3": (1664, 1248), "3,4": (1248, 1664), "16,9": (1920, 1080), "9,16": (1080, 1920)},
    "1280": {"1,1": (1712, 1712), "4,3": (1968, 1472), "3,4": (1472, 1968), "16,9": (2272, 1280), "9,16": (1280, 2272)},
    "2048": {
        "1,1": (2728, 2728),
        "4,3": (3160, 2368),
        "3,4": (2368, 3160),
        "16,9": (3640, 2048),
        "9,16": (2048, 3640),
    },
    "gt_2048": {
        "1,1": (5464, 5464),
        "4,3": (6304, 4728),
        "3,4": (4728, 6304),
        "16,9": (7280, 4096),
        "9,16": (4096, 7280),
    },
    # 512 square tier -- gripperhead FDM at native LIBERO 512x512 (nomask). latent
    # side = 512/(16*2) = 16, within the default rope grid (20), so nomask needs no
    # model change; the mask 512_w2 tier below (latent w=32) does.
    "512": {"1,1": (512, 512), "4,3": (640, 512), "3,4": (512, 640), "16,9": (640, 384), "9,16": (384, 640)},
    # 384 square tier -- gripperhead FDM at 384x384 (nomask). latent side = 384/(16*2) = 12, within
    # the default rope grid; the mask 384_w2 tier below (latent w=24) needs max_vae_latent_side>=24.
    "384": {"1,1": (384, 384), "4,3": (480, 384), "3,4": (384, 480), "16,9": (480, 288), "9,16": (288, 480)},
    # --- side-by-side "RGB | mask" layouts (2:1 wide, single bucket). Used only
    # by the gripperhead FDM `use_mask` option, which emits a square RGB frame
    # concatenated with its square mask frame -> aspect 2:1. Kept as dedicated
    # tiers (not extra buckets on "256"/"480") so aspect-bucket selection for all
    # other data -- e.g. DROID multi-view concat videos -- is left untouched.
    "256_w2": {"2,1": (512, 256)},   # -> 256h x 512w
    "480_w2": {"2,1": (1280, 640)},  # -> 640h x 1280w
    "512_w2": {"2,1": (1024, 512)},  # -> 512h x 1024w (native LIBERO; latent w=1024/32=32 -> needs max_vae_latent_side_after_patchify>=32)
    "384_w2": {"2,1": (768, 384)},   # -> 384h x 768w (gripperhead mask; latent w=768/32=24 -> needs max_vae_latent_side_after_patchify>=24)
}


def get_aspect_ratios_from_wdinfos(wdinfos: list[str]) -> list[str]:
    aspect_ratios = []
    for wdinfo in wdinfos:
        aspect_ratio_match = re.search(r"aspect_ratio_(\d+_\d+)", wdinfo)
        aspect_ratios.append(aspect_ratio_match.group(1))

    return aspect_ratios


def get_wdinfos_w_aspect_ratio(wdinfos: list[str]) -> List[Tuple[str, str]]:
    aspect_ratios = get_aspect_ratios_from_wdinfos(wdinfos)

    # return a list of (wdinfo_path, aspect_ratio) pairs
    return [(wdinfo, aspect_ratio.replace("_", ",")) for wdinfo, aspect_ratio in zip(wdinfos, aspect_ratios)]


def parse_frame_range_from_wdinfo(wdinfo: str) -> tuple[int, int] | None:
    """
    Parse frame range from wdinfo path.

    Args:
        wdinfo: wdinfo path string containing frames_X_Y pattern

    Returns:
        Tuple of (min_frames, max_frames) if found, None otherwise

    Example:
        >>> parse_frame_range_from_wdinfo("wdinfo/v4/tv_drama/resolution_720/aspect_ratio_16_9/frames_300_400/wdinfo.json")
        (300, 400)
    """
    match = re.search(r"frames_(\d+)_(\d+)", wdinfo)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def filter_wdinfos_by_frame_range(
    wdinfos: list[str],
    min_frames: int | None = None,
    max_frames: int | None = None,
) -> list[str]:
    """
    Filter wdinfo files based on frame range.

    The frame range in wdinfo path (e.g., frames_300_400) represents videos
    with frames between those values. This function filters wdinfo files
    based on the wdinfo's upper bound (wdinfo_max):
    - min_frames is EXCLUSIVE: wdinfo_max must be > min_frames
    - max_frames is INCLUSIVE: wdinfo_max must be <= max_frames

    Args:
        wdinfos: List of wdinfo paths
        min_frames: Minimum number of frames (exclusive). If None, no lower bound.
        max_frames: Maximum number of frames (inclusive). If None, no upper bound.

    Returns:
        Filtered list of wdinfo paths

    Example:
        >>> wdinfos = [
        ...     "wdinfo/frames_400_500/wdinfo.json",
        ...     "wdinfo/frames_500_600/wdinfo.json",
        ...     "wdinfo/frames_600_700/wdinfo.json",
        ... ]
        >>> filter_wdinfos_by_frame_range(wdinfos, min_frames=500, max_frames=600)
        ['wdinfo/frames_500_600/wdinfo.json']
        # frames_400_500 excluded because wdinfo_max (500) <= min_frames (500)
        # frames_500_600 included because wdinfo_max (600) > min_frames (500) AND <= max_frames (600)
        # frames_600_700 excluded because wdinfo_max (700) > max_frames (600)
    """
    if min_frames is None and max_frames is None:
        return wdinfos

    filtered = []
    for wdinfo in wdinfos:
        frame_range = parse_frame_range_from_wdinfo(wdinfo)
        if frame_range is None:
            # If no frame range in path, include by default
            filtered.append(wdinfo)
            continue

        wdinfo_min, wdinfo_max = frame_range

        # Filter based on wdinfo's upper bound (wdinfo_max):
        # - min_frames is exclusive: wdinfo_max must be > min_frames
        # - max_frames is inclusive: wdinfo_max must be <= max_frames
        include = True
        if min_frames is not None and wdinfo_max <= min_frames:
            include = False
        if max_frames is not None and wdinfo_max > max_frames:
            include = False

        if include:
            filtered.append(wdinfo)

    return filtered
