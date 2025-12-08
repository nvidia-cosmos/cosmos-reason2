# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import numpy as np
import pydantic
import torch
from PIL import Image
from qwen_vl_utils.vision_process import SPATIAL_MERGE_SIZE as SPATIAL_MERGE_SIZE

"""Vision processing utilities."""

# https://huggingface.co/nvidia/Cosmos-Reason2-2B/blob/main/video_preprocessor_config.json#L6
IMAGE_PATCH_SIZE = 16
PATCH_FACTOR = IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE
PIXELS_PER_TOKEN = PATCH_FACTOR**2
"""Number of pixels per visual token."""


class VisionConfig(pydantic.BaseModel):
    """Config for vision processing.

    Source:
    https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py

    Attributes are sorted by priority. Higher priority attributes override lower
    priority attributes.
    """

    model_config = pydantic.ConfigDict(extra="allow", use_attribute_docstrings=True)

    resized_height: int | None = None
    """Max height of the image/video"""
    resized_width: int | None = None
    """Max width of the image/video"""

    min_pixels: int | None = None
    """Min frame pixels of the image/video"""
    max_pixels: int | None = None
    """Max frame pixels of the image/video"""
    total_pixels: int | None = None
    """Max total pixels of the image/video"""

    video_start: float | None = None
    """Start time of the video (seconds)"""
    video_end: float | None = None
    """End time of the video (seconds)"""

    nframes: int | None = None
    """Number of frames of the video"""

    fps: float | None = None
    """FPS of the video"""
    min_frames: int | None = None
    """Min frames of the video"""
    max_frames: int | None = None
    """Max frames of the video"""


def _tensor_to_pil_images(tensor: torch.Tensor) -> list[Image.Image]:
    """Convert a tensor to a list of PIL images.

    Args:
        tensor: Tensor with shape (C, H, W), (C, T, H, W) or (T, C, H, W)

    Returns:
        List of PIL images
    """
    # Check tensor shape and convert if needed
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.shape[0] == 3:  # (C, T, H, W)
        if tensor.shape[1] == 3:
            raise ValueError(f"Ambiguous shape: {tensor.shape}")
        # Convert to (T, C, H, W)
        tensor = tensor.permute(1, 0, 2, 3)

    # Convert to numpy array with shape (T, H, W, C)
    frames = tensor.permute(0, 2, 3, 1).cpu().numpy()

    # Ensure values are in the right range for PIL (0-255, uint8)
    if np.issubdtype(frames.dtype, np.floating):
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    return [Image.fromarray(frame) for frame in frames]


def save_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a tensor as images to a directory.

    Args:
        tensor: Tensor with shape (C, H, W) or (T, C, H, W)
        path: Directory to save the images
    """
    os.makedirs(path, exist_ok=True)
    images = _tensor_to_pil_images(tensor)
    for i, image in enumerate(images):
        image.save(f"{path}/{i}.png")
