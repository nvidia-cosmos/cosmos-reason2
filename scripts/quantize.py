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

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "llmcompressor @ git+https://github.com/vllm-project/llm-compressor.git@6e459ed",
#   "pillow>=2.2.1",
#   "pydantic>=2.12.4",
#   "qwen-vl-utils==0.0.14",
#   "torch==2.8.0",
#   "torchcodec>=0.8.1",
#   "torchvision",
#   "tyro>=0.9.35",
#   "transformers @ git+https://github.com/huggingface/transformers.git@def9a7ef057b13d04aeeaa150e3ce63afa151d4e",
# ]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128"}]
# torchvision = [{ index = "pytorch-cu128"}]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

"""Quantize a Cosmos-Reason2 model.

Example:

```shell
./scripts/quantize.py
```
"""

import os
from typing import Annotated, Literal

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import base64
import json
import shutil
from io import BytesIO
from pathlib import Path

import pydantic
import requests
import torch
import tyro
from datasets import load_dataset
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.utils import dispatch_for_generation
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

Precision = Literal["nvfp4", "fp8", "fp8_dynamic"]
KvPrecision = Literal["bf16", "fp8"]


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Directory to save the quantized model. Model will be saved in {output_dir}/model_{precision}."""
    model: str = "nvidia/Cosmos-Reason2-2B"
    """Local path to a model or model name from https://huggingface.co/collections/nvidia/cosmos-reason2."""
    num_samples: int = 512
    """Number of samples to use for calibration."""
    precision: Precision = "nvfp4"
    """Precision to use for quantization."""
    kv_precision: KvPrecision = "bf16"
    """Precision to use for the KV cache quantization."""
    smoothing_strength: float = 0.8
    """Smoothing strength to use for SmoothQuant."""
    max_sequence_length: int = 262144
    """Maximum sequence length to use for quantization. (Defaults to CR2/Qwen3-VL max model len)"""
    seed: int = 42
    """Seed to use for random number generator."""


def preprocess_and_tokenize(
    example: dict, processor: AutoProcessor, max_sequence_length: int
) -> dict:
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_qwen},
                {"type": "text", "text": "What does the image show?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
    )


def data_collator(batch: list[dict]) -> dict:
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def get_quantization_recipe(
    precision: Precision, kv_precision: KvPrecision, smoothing_strength: float
) -> list[SmoothQuantModifier | QuantizationModifier]:
    kv_scheme = (
        None
        if kv_precision == "bf16"
        else {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False}
    )
    recipe = [
        SmoothQuantModifier(
            smoothing_strength=smoothing_strength,
            mappings=[
                [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
                [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
            ],
        ),
        QuantizationModifier(
            targets="Linear",
            scheme=precision.upper(),
            ignore=[
                "re:.*lm_head",
                "re:visual.*",
                "re:model.visual.*",
                "re:.*mlp.gate$",
            ],
            kv_cache_scheme=kv_scheme,
        ),
    ]
    return recipe


def run_sample_generation(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    max_sequence_length: int,
):
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    test_url = "http://images.cocodataset.org/train2017/000000231895.jpg"
    test_image = Image.open(BytesIO(requests.get(test_url).content))

    test_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_url},
                {"type": "text", "text": "Please describe the animal in this image\n"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(test_messages, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[test_image],
        padding=False,
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    print("Generating response...")
    output = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    print("==========================================")


def save_model(
    model: Qwen3VLForConditionalGeneration, processor: AutoProcessor, output_dir: Path
):
    model.save_pretrained(output_dir, save_compressed=True)


def postprocess_config(config_path: Path):
    def remove_keys(d, keys_to_remove):
        if isinstance(d, dict):
            return {
                k: remove_keys(v, keys_to_remove)
                for k, v in d.items()
                if k not in keys_to_remove
            }
        elif isinstance(d, list):
            return [remove_keys(i, keys_to_remove) for i in d]
        else:
            return d

    with open(config_path) as f:
        config = json.load(f)
    clean_config = remove_keys(config, keys_to_remove=["zp_dtype", "scale_dtype"])
    with open(config_path, "w") as f:
        json.dump(clean_config, f, indent=2)


def quantize(args: Args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = replace_modules_for_calibration(model)
    dataset_id = "lmms-lab/flickr30k"
    dataset_split = {"calibration": f"test[:{args.num_samples}]"}
    output_dir = Path(args.output_dir) / f"model_{args.precision}"
    sequential_targets = ["Qwen3VLTextDecoderLayer"]

    # workaround to register KV metadata to the VLM HF config
    model.config.num_attention_heads = model.config.text_config.num_attention_heads
    model.config.num_key_value_heads = model.config.text_config.num_key_value_heads
    model.config.head_dim = model.config.text_config.head_dim

    print(f"Loading calibration dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=args.seed)
    print("Preprocessing dataset...")
    ds = ds.map(
        lambda x: preprocess_and_tokenize(x, processor, args.max_sequence_length),
        batched=False,
        remove_columns=ds["calibration"].column_names,
    )
    recipe = get_quantization_recipe(
        args.precision, args.kv_precision, args.smoothing_strength
    )

    print(f"Starting {args.precision} quantization process...")
    oneshot(
        model=model,
        recipe=recipe,
        max_seq_length=args.max_sequence_length,
        num_calibration_samples=args.num_samples,
        dataset=ds,
        data_collator=data_collator,
        sequential_targets=sequential_targets,
    )
    print("Quantization complete!")
    print("Running sample generation...")
    run_sample_generation(model, processor, args.max_sequence_length)
    print(f"Saving quantized model to: {output_dir}...")
    save_model(model, processor, output_dir)
    config_path = output_dir / "config.json"
    print(f"Postprocessing config file {config_path}...")
    postprocess_config(config_path)
    if not (model_path := Path(args.model)).exists():
        # path for remote model / HF ID
        snapshot_download(
            repo_id=args.model,
            ignore_patterns=["config.json", "*.safetensors*"],
            local_dir=output_dir,
        )
    else:
        # path for local model directory
        files_to_copy = [
            f
            for f in model_path.glob("*")
            if f.name != "config.json"
            and "safetensors" not in f.name
            and not f.is_dir()
        ]
        for file in files_to_copy:
            shutil.copy(file, output_dir / file.name)
    print(f"Quantization complete! Model saved to: {output_dir}")


def main():
    args = tyro.cli(Args, description=__doc__)
    quantize(args)


if __name__ == "__main__":
    main()
