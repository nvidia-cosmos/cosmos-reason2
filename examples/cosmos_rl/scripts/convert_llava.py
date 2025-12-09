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

"""Convert LLaVA dataset to cosmos-rl format."""

import json
import re
from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_reason2_utils.text import create_conversation
from cosmos_reason2_utils.vision import VisionConfig
from rich import print
from tqdm import tqdm


class Args(pydantic.BaseModel):
    input_file: Annotated[pydantic.FilePath, tyro.conf.arg(aliases=("-i",))] = (
        tyro.MISSING
    )
    """Input annotations json file path."""
    output_file: Annotated[Path, tyro.conf.arg(aliases=("-o",))] = tyro.MISSING
    """Output dataset jsonl file path."""
    media: pydantic.DirectoryPath | None = None
    """Media directory path."""

    system_prompt: str | None = None
    """System prompt."""
    vision: VisionConfig = VisionConfig()
    """Vision processor config."""

    num_samples: int | None = None
    """Truncate number of samples."""


def convert_llava(args: Args):
    annotations = json.load(args.input_file.open())
    if args.num_samples is not None:
        annotations = annotations[: args.num_samples]
    num_samples = len(annotations)
    print(f"Number of samples: {num_samples}")

    def process_sample(sample: dict) -> str:
        user_prompt = sample["conversations"][0]["value"]
        response = sample["conversations"][1]["value"]
        images = sample.get("image", None) or sample.get("images", None)
        if images and isinstance(images, str):
            images = [images]
        videos = sample.get("video", None)
        if videos and isinstance(videos, str):
            videos = [videos]

        # Prepend media directory path
        if args.media is not None:
            if images:
                images = [f"{args.media}/img" for img in images]
            if videos:
                videos = [f"{args.media}/vid" for vid in videos]

        # cosmos-rl expects base64 encoded images
        # for i, image in enumerate(images):
        #     images[i] = base64.b64encode(open(image, "rb").read())

        # Remove image and video tags from user prompt
        user_prompt = re.sub(r"(\n)?</?(image|video)>(\n)?", "", user_prompt)

        messages = create_conversation(
            system_prompt=args.system_prompt,
            user_prompt=user_prompt,
            response=response,
            images=images,
            videos=videos,
            vision_kwargs=args.vision.model_dump(exclude_none=True),
        )
        return json.dumps(
            {
                "messages": messages,
            }
        )

    dataset = list(
        tqdm(
            map(process_sample, annotations),
            total=num_samples,
            desc="Processing samples",
        )
    )
    print(dataset[0])
    args.output_file.write_text("\n".join(dataset))


def main():
    args = tyro.cli(Args)
    convert_llava(args)


if __name__ == "__main__":
    main()
