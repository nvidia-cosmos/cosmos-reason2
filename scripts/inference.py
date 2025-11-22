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

"""Full example of inference with Cosmos-Reason2.

Example:

```shell
uv run scripts/inference.py --prompt prompts/caption.yaml --videos assets/sample.mp4 -v
```
"""

# Source: https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#offline-inference

from cosmos_reason_utils.script import init_script

init_script()

import collections
import pathlib
import textwrap
from typing import Annotated

import pydantic
import qwen_vl_utils
import transformers
import tyro
import vllm
import yaml
from cosmos_reason_utils.text import (
    PromptConfig,
    create_conversation,
    extract_tagged_text,
)
from cosmos_reason_utils.vision import (
    VisionConfig,
    overlay_text_on_tensor,
    save_tensor,
)
from rich import print
from rich.pretty import pprint

ROOT = pathlib.Path(__file__).parents[1].resolve()
SEPARATOR = "-" * 20


def pprint_dict(d: dict, name: str):
    """Pretty print a dictionary."""
    pprint(collections.namedtuple(name, d.keys())(**d), expand_all=True)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    prompt: pydantic.FilePath
    """Path to prompt yaml file"""

    images: list[str] = pydantic.Field(default_factory=list)
    """Image paths"""
    videos: list[str] = pydantic.Field(default_factory=list)
    """Video paths"""
    timestamp: bool = False
    """Overlay timestamp on video frames"""
    question: str | None = None
    """Question to ask the model (user prompt)"""
    reasoning: bool = False
    """Enable reasoning trace"""
    vision_config: pydantic.FilePath = ROOT / "configs/vision_config.yaml"
    """Path to vision config yaml file"""
    sampling_params: pydantic.FilePath = ROOT / "configs/sampling_params.yaml"
    """Path to sampling parameters yaml file"""
    model: str = "nvidia/Cosmos-Reason2-2B-v1.0"
    """Model name or path (Cosmos-Reason2: https://huggingface.co/collections/nvidia/cosmos-reason2)"""
    revision: str | None = None
    """Model revision (branch name, tag name, or commit id)"""
    verbose: Annotated[bool, tyro.conf.arg(aliases=("-v",))] = False
    """Verbose output"""
    output: Annotated[str | None, tyro.conf.arg(aliases=("-o",))] = None
    """Output directory for debugging"""


def main(args: Args):
    images: list[str] = args.images or []
    videos: list[str] = args.videos or []

    # Load configs
    prompt_kwargs = yaml.safe_load(open(args.prompt, "rb"))
    prompt_config = PromptConfig.model_validate(prompt_kwargs)
    vision_kwargs = yaml.safe_load(open(args.vision_config, "rb"))
    _vision_config = VisionConfig.model_validate(vision_kwargs)
    sampling_kwargs = yaml.safe_load(open(args.sampling_params, "rb"))
    sampling_params = vllm.SamplingParams(**sampling_kwargs)
    if args.verbose:
        pprint_dict(vision_kwargs, "VisionConfig")
        pprint_dict(sampling_kwargs, "SamplingParams")

    # Create conversation
    system_prompts = [open(f"{ROOT}/prompts/addons/english.txt").read()]
    if prompt_config.system_prompt:
        system_prompts.append(prompt_config.system_prompt)
    if args.reasoning and "<think>" not in prompt_config.system_prompt:
        if extract_tagged_text(prompt_config.system_prompt)[0]:
            raise ValueError(
                "Prompt already contains output format. Cannot add reasoning."
            )
        system_prompts.append(open(f"{ROOT}/prompts/addons/reasoning.txt").read())
    # pyrefly: ignore [no-matching-overload]
    system_prompt = "\n\n".join(map(str.rstrip, system_prompts))
    if args.question:
        user_prompt = args.question
    else:
        user_prompt = prompt_config.user_prompt
    if not user_prompt:
        raise ValueError("No user prompt provided.")
    user_prompt = user_prompt.rstrip()
    conversation = create_conversation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        videos=videos,
        vision_kwargs=vision_kwargs,
    )
    if args.verbose:
        pprint(conversation, expand_all=True)
    print(SEPARATOR)
    print("System:")
    print(textwrap.indent(system_prompt.rstrip(), "  "))
    print("User:")
    print(textwrap.indent(user_prompt.rstrip(), "  "))
    print(SEPARATOR)

    # Create model
    llm = vllm.LLM(
        model=args.model,
        revision=args.revision,
        limit_mm_per_prompt={"image": len(images), "video": len(videos)},
        enforce_eager=True,
    )

    # Process inputs
    processor: transformers.Qwen3VLProcessor = (
        # pyrefly: ignore [bad-assignment]
        transformers.AutoProcessor.from_pretrained(args.model)
    )
    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation,
        # pyrefly: ignore [missing-attribute]
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if args.timestamp:
        video_inputs = [
            # pyrefly: ignore [bad-argument-type, bad-index]
            (overlay_text_on_tensor(video, fps=video_metadata["fps"]), video_metadata)
            # pyrefly: ignore [not-iterable]
            for video, video_metadata in video_inputs
        ]
    if args.output:
        if image_inputs is not None:
            for i, image in enumerate(image_inputs):
                # pyrefly: ignore [bad-argument-type]
                save_tensor(image, f"{args.output}/image_{i}")
        if video_inputs is not None:
            # pyrefly: ignore [bad-argument-type]
            for i, (video, _) in enumerate(video_inputs):
                # pyrefly: ignore [bad-argument-type]
                save_tensor(video, f"{args.output}/video_{i}")

    # Run inference
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        # pyrefly: ignore [unsupported-operation]
        mm_data["video"] = video_inputs
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    # pyrefly: ignore [bad-argument-type]
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    print(SEPARATOR)
    for output in outputs[0].outputs:
        output_text = output.text
        print("Assistant:")
        print(textwrap.indent(output_text.rstrip(), "  "))
    print(SEPARATOR)


if __name__ == "__main__":
    args = tyro.cli(Args, description=__doc__)
    main(args)
