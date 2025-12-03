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

"""Minimal example of inference with Cosmos-Reason2.

Example:

```shell
uv run scripts/inference_sample.py
```
"""

# Source: https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#new-qwen-vl-utils-usage

from pathlib import Path

import torch
import transformers
from qwen_vl_utils import process_vision_info

ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 20


def main():
    # Load model
    model_name = "nvidia/Cosmos-Reason2-2B"
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    processor: transformers.Qwen3VLProcessor = (
        # pyrefly: ignore [bad-assignment]
        transformers.AutoProcessor.from_pretrained(model_name)
    )

    # Create inputs
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"{ROOT}/assets/sample.mp4",
                    "fps": 4,
                    # 6422528 = 8192 * 28**2 = vision_tokens * (2*spatial_patch_size)^2
                    "total_pixels": 6422528,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    images, videos, video_kwargs = process_vision_info(
        conversation,
        # pyrefly: ignore [missing-attribute]
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadatas = zip(*videos, strict=False)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(
        text=text,
        # pyrefly: ignore [bad-argument-type]
        images=images,
        # pyrefly: ignore [bad-argument-type]
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        # pyrefly: ignore [bad-unpacking]
        **video_kwargs,
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
