#!/usr/bin/env -S uv run --script
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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=1.12.0",
#   "pillow>=12.0.0",
#   "transformers>=4.57.0",
#   "torch==2.8.0",
#   "torchvision",
#   "torchcodec==0.7.0",
# ]
# ///

"""Minimal example of inference with Cosmos-Reason2."""

# Source: https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#new-qwen-vl-utils-usage

from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 20

PIXELS_PER_TOKEN = 32**2
"""Number of pixels per visual token."""


def main():
    # Load model
    model_name = "nvidia/Cosmos-Reason2-2B"
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    processor: transformers.Qwen3VLProcessor = (
        transformers.AutoProcessor.from_pretrained(model_name)
    )

    # Optional: Limit vision tokens
    min_vision_tokens = 256
    max_vision_tokens = 8192
    processor.image_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }
    processor.video_processor.size = {
        "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
        "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
    }

    # Create inputs
    # IMPORTANT: Media is listed before text to match training inputs
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"{ROOT}/assets/sample.mp4",
                },
                {"type": "text", "text": "Caption the video in detail."},
            ],
        },
    ]

    # Process inputs
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=4,
    )
    inputs = inputs.to(model.device)

    # Run inference
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(SEPARATOR)
    print(output_text[0])
    print(SEPARATOR)


if __name__ == "__main__":
    main()
