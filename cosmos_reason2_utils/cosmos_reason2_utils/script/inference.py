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

"""Inference using vLLM."""

# Sources
# * https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#deployment

import time
from cosmos_reason2_utils.init import init_script

init_script()

import collections
import textwrap
from functools import cached_property
from typing import Annotated

import openai
import pydantic
import qwen_vl_utils
import transformers
import tyro
import vllm
import yaml
from rich import print
from rich.pretty import pprint
from typing_extensions import assert_never

from cosmos_reason2_utils.text import (
    REASONING_PROMPT,
    SYSTEM_PROMPT,
    create_conversation,
    create_conversation_openai,
)
from cosmos_reason2_utils.vision import (
    PIXELS_PER_TOKEN,
    VisionConfig,
    save_tensor,
)

SEPARATOR = "-" * 20

DEFAULT_MODEL = "nvidia/Cosmos-Reason2-2B"
"""Default model name."""


def pprint_dict(d: dict, name: str):
    """Pretty print a dictionary."""
    pprint(collections.namedtuple(name, d.keys())(**d), expand_all=True)


# Copied from [vllm.SamplingParams](https://docs.vllm.ai/en/latest/dev/sampling_params.html).
# Ideally, we could auto-generate this class.
class SamplingOverrides(pydantic.BaseModel):
    """Sampling parameters for text generation."""

    model_config = pydantic.ConfigDict(extra="allow", use_attribute_docstrings=True)

    n: int | None = None
    """Number of outputs to return for the given prompt request."""
    presence_penalty: float | None = None
    """Penalizes new tokens based on whether they appear in the generated text
    so far. Values > 0 encourage the model to use new tokens, while values < 0
    encourage the model to repeat tokens."""
    repetition_penalty: float | None = None
    """Penalizes new tokens based on whether they appear in the prompt and the
    generated text so far. Values > 1 encourage the model to use new tokens,
    while values < 1 encourage the model to repeat tokens."""
    temperature: float | None = None
    """Controls the randomness of the sampling. Lower values make the model
    more deterministic, while higher values make the model more random. Zero
    means greedy sampling."""
    top_p: float | None = None
    """Controls the cumulative probability of the top tokens to consider. Must
    be in (0, 1]. Set to 1 to consider all tokens."""
    top_k: int | None = None
    """Controls the number of top tokens to consider. Set to 0 (or -1) to
    consider all tokens."""
    seed: int | None = None
    """Random seed to use for the generation."""
    max_tokens: int | None = None
    """Maximum number of tokens to generate per output sequence."""

    @classmethod
    def get_defaults(cls, *, reasoning: bool = False) -> dict:
        kwargs = dict(
            max_tokens=4096,
        )
        # Source: https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#evaluation-reproduction
        if reasoning:
            return kwargs | dict(
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.0,
                presence_penalty=0.0,
                temperature=0.6,
                seed=1234,
            )
        else:
            return kwargs | dict(
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.0,
                presence_penalty=1.5,
                temperature=0.7,
                seed=3407,
            )


class InputConfig(pydantic.BaseModel):
    """Prompt config."""

    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    user_prompt: str = ""
    """User prompt."""
    system_prompt: str = pydantic.Field(default=SYSTEM_PROMPT)
    """System prompt."""
    sampling_params: dict = pydantic.Field(default_factory=dict)
    """Override sampling parameters."""


class Args(pydantic.BaseModel):
    """Inference arguments."""

    model_config = pydantic.ConfigDict(
        extra="forbid", use_attribute_docstrings=True, frozen=True
    )

    input_file: Annotated[pydantic.FilePath | None, tyro.conf.arg(aliases=("-i",))] = (
        None
    )
    """Path to input yaml file."""
    prompt: str | None = None
    """User prompt."""
    images: list[str] = pydantic.Field(default_factory=list)
    """Image paths or URLs."""
    videos: list[str] = pydantic.Field(default_factory=list)
    """Video paths or URLs."""
    reasoning: bool = False
    """Enable reasoning trace."""
    sampling: SamplingOverrides = SamplingOverrides()
    """Sampling parameters."""

    verbose: Annotated[bool, tyro.conf.arg(aliases=("-v",))] = False
    """Verbose output"""

    @cached_property
    def input_config(self) -> InputConfig:
        if self.input_file is not None:
            input_kwargs = yaml.safe_load(open(self.input_file, "rb"))
            return InputConfig.model_validate(input_kwargs)
        else:
            return InputConfig()

    @cached_property
    def system_prompt(self) -> str:
        return self.input_config.system_prompt

    @cached_property
    def user_prompt(self) -> str:
        if self.prompt:
            user_prompt = self.prompt
        else:
            user_prompt = self.input_config.user_prompt
        if not user_prompt:
            raise ValueError("No user prompt provided.")
        user_prompt = user_prompt.strip()
        if self.reasoning:
            user_prompt += f"\n\n{REASONING_PROMPT}"
        return user_prompt

    @cached_property
    def sampling_kwargs(self) -> dict:
        sampling_kwargs = SamplingOverrides.get_defaults(reasoning=self.reasoning)
        sampling_kwargs.update(self.input_config.sampling_params)
        sampling_kwargs.update(self.sampling.model_dump(exclude_none=True))
        vllm.SamplingParams(**sampling_kwargs)
        return sampling_kwargs

    @cached_property
    def sampling_params(self) -> vllm.SamplingParams:
        return vllm.SamplingParams(**self.sampling_kwargs)


class Offline(Args):
    """Offline inference arguments."""

    model: str = DEFAULT_MODEL
    """Model name or path (https://huggingface.co/collections/nvidia/cosmos-reason2)."""
    revision: str | None = None
    """Model revision (branch name, tag name, or commit id)."""
    max_model_len: int = qwen_vl_utils.vision_process.MODEL_SEQ_LEN
    """Maximum model length.
    
    If specified, input media will be resized to fit in the model length.
    """

    vision: VisionConfig = VisionConfig()
    """Vision processor config."""

    output: Annotated[str | None, tyro.conf.arg(aliases=("-o",))] = None
    """Output directory for debugging."""


class Online(Args):
    """Online inference arguments."""

    host: str = "localhost"
    """Server hostname."""
    port: int = 8000
    """Server port."""
    model: str | None = None
    """Model name (https://huggingface.co/collections/nvidia/cosmos-reason2).
    
    If not provided, the first model in the server will be used.
    """

    fps: float | None = None
    """FPS of the video."""
    min_pixels: int | None = None
    """Min total pixels of the image/video."""
    total_pixels: int | None = None
    """Max total pixels of the image/video."""


def offline_inference(args: Offline):
    # Limit total pixels to fit in model length
    vision_kwargs = args.vision.model_dump(exclude_none=True)
    assert args.sampling_params.max_tokens
    if args.max_model_len < args.sampling_params.max_tokens:
        raise ValueError("Max model length must be greater than max tokens.")
    max_seq_len = args.max_model_len - args.sampling_params.max_tokens
    total_pixels = int(max_seq_len * PIXELS_PER_TOKEN * 0.9)
    if "total_pixels" in vision_kwargs:
        if vision_kwargs["total_pixels"] > total_pixels:
            raise ValueError(
                f"Total pixels {vision_kwargs['total_pixels']} exceeds limit {total_pixels}."
            )
    else:
        vision_kwargs["total_pixels"] = total_pixels
    VisionConfig.model_validate(vision_kwargs)
    if args.verbose:
        pprint_dict(vision_kwargs, "VisionConfig")

    conversation = create_conversation(
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        images=args.images,
        videos=args.videos,
        vision_kwargs=vision_kwargs,
    )
    if args.verbose:
        pprint(conversation, expand_all=True)

    # Create model
    llm = vllm.LLM(
        model=args.model,
        revision=args.revision,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": len(args.images), "video": len(args.videos)},
    )

    # Process inputs
    processor: transformers.Qwen3VLProcessor = (
        transformers.AutoProcessor.from_pretrained(args.model)
    )
    add_vision_ids = (len(args.images) + len(args.videos)) > 1
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_ids=add_vision_ids,
    )
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        conversation,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if args.output:
        if image_inputs is not None:
            for i, image in enumerate(image_inputs):
                save_tensor(image, f"{args.output}/image_{i}")
        if video_inputs is not None:
            for i, (video, _) in enumerate(video_inputs):
                save_tensor(video, f"{args.output}/video_{i}")

    # Run inference
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    outputs = llm.generate([llm_inputs], sampling_params=args.sampling_params)
    for output in outputs[0].outputs:
        if args.verbose:
            pprint(output, expand_all=True)
        print(SEPARATOR)
        print("Assistant:")
        print(textwrap.indent(output.text.strip(), "  "))
        print(SEPARATOR)


def online_inference(args: Online):
    # Create client
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.host}:{args.port}/v1",
    )
    models = client.models.list()
    if args.model is not None:
        model = client.models.retrieve(args.model)
    else:
        model = models.data[0]
    if args.verbose:
        pprint(model, expand_all=True)

    # Create conversation
    conversation = create_conversation_openai(
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        images=args.images,
        videos=args.videos,
    )
    if args.verbose:
        pprint(conversation, expand_all=True)

    # Process inputs
    max_model_len = model.max_model_len or qwen_vl_utils.vision_process.MODEL_SEQ_LEN
    assert args.sampling_params.max_tokens
    if max_model_len < args.sampling_params.max_tokens:
        raise ValueError("Max model length must be greater than max tokens.")
    max_seq_len = max_model_len - args.sampling_params.max_tokens
    total_pixels = int(max_seq_len * PIXELS_PER_TOKEN * 0.9)
    if args.total_pixels:
        if args.total_pixels > total_pixels:
            raise ValueError(
                f"Total pixels {args.total_pixels} exceeds limit {total_pixels}."
            )
        total_pixels = args.total_pixels
    mm_processor_kwargs = {
        "fps": args.fps or qwen_vl_utils.vision_process.FPS,
        "do_sample_frames": True,
        "size": {
            "shortest_edge": args.min_pixels
            or qwen_vl_utils.vision_process.VIDEO_MIN_TOKEN_NUM * PIXELS_PER_TOKEN,
            "longest_edge": total_pixels,
        },
    }
    if args.verbose:
        pprint_dict(mm_processor_kwargs, "mm_processor_kwargs")

    # Run inference
    chat_completion = client.chat.completions.create(
        messages=conversation,
        model=model.id,
        extra_body=args.sampling_kwargs | {"mm_processor_kwargs": mm_processor_kwargs},
    )

    for output in chat_completion.choices:
        if args.verbose:
            pprint(output, expand_all=True)
        if output.message.reasoning_content:
            print(SEPARATOR)
            print("Reasoning:")
            print(textwrap.indent(output.message.reasoning_content.strip(), "  "))
        print(SEPARATOR)
        print("Assistant:")
        print(textwrap.indent(output.message.content.strip(), "  "))
        print(SEPARATOR)


def inference(args: Offline | Online):
    print(SEPARATOR)
    print("System:")
    print(textwrap.indent(args.system_prompt.strip(), "  "))
    print(SEPARATOR)
    print("User:")
    print(textwrap.indent(args.user_prompt.strip(), "  "))
    print(SEPARATOR)
    if args.verbose:
        pprint_dict(args.sampling_kwargs, "SamplingParams")

    start = time.time()
    if isinstance(args, Offline):
        offline_inference(args)
    elif isinstance(args, Online):
        online_inference(args)
    else:
        assert_never(args)
    duration = time.time() - start
    print(f"Inference time: {duration:.2f} seconds")


def main():
    args = tyro.cli(
        Offline | Online, description=__doc__, config=(tyro.conf.OmitArgPrefixes,)
    )
    inference(args)


if __name__ == "__main__":
    main()
