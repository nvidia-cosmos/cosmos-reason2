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
#   "qwen-vl-utils==0.0.14",
#   "torch==2.8.0",
#   "torchcodec>=0.8.1",
#   "torchvision",
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
./scripts/quantize.py --model nvidia/Cosmos-Reason2-2B --save_dir checkpoints/Cosmos-Reason2-2B-FP4
```
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import requests
import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.utils import dispatch_for_generation
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        help="Should be `nvidia/Cosmos-Reason2-2B`, `nvidia/Cosmos-Reason2-8B` or local path to a model.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Directory to save the quantized model. Model will be saved in {save_dir}/model_{precision}.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=512,
        help="Number of samples to use for calibration.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp4",
        help="Precision to use for quantization.",
        choices=["fp4", "fp8"],
    )
    parser.add_argument(
        "--smoothing_strength",
        type=float,
        default=0.8,
        help="Smoothing strength to use for SmoothQuant.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=32768,
        help="Maximum sequence length to use for quantization.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to use for random number generator.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() and args.model not in [
        "nvidia/Cosmos-Reason2-2B",
        "nvidia/Cosmos-Reason2-8B",
    ]:
        error_message = f"Model path {args.model} is not a valid model. Valid models are: nvidia/Cosmos-Reason2-2B, nvidia/Cosmos-Reason2-8B or local path to a model."
        raise ValueError(error_message)
    save_dir_path = Path(args.save_dir)
    if not save_dir_path.exists():
        print("Saving directory does not exist. Creating...")
        save_dir_path.mkdir(parents=True, exist_ok=True)
    return args


def preprocess_and_tokenize(example, processor, max_sequence_length):
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


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def get_quantization_recipe(precision, smoothing_strength):
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
            scheme=precision,
            ignore=[
                "re:.*lm_head",
                "re:visual.*",
                "re:model.visual.*",
                "re:.*mlp.gate$",
            ],
        ),
    ]
    return recipe


def run_sample_generation(model, processor, max_sequence_length):
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


def save_model(model, processor, save_dir):
    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)


def postprocess_config(config_path):
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


def main(args):
    model_path = Path(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model = replace_modules_for_calibration(model)
    dataset_id = "lmms-lab/flickr30k"
    dataset_split = {"calibration": f"test[:{args.num_samples}]"}
    precision = "NVFP4" if args.precision == "fp4" else "FP8_DYNAMIC"
    save_dir = Path(args.save_dir) / f"model_{args.precision}"
    sequential_targets = ["Qwen3VLTextDecoderLayer"]

    print(f"Loading calibration dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=args.seed)
    print("Preprocessing dataset...")
    ds = ds.map(
        lambda x: preprocess_and_tokenize(x, processor, args.max_sequence_length),
        batched=False,
        remove_columns=ds["calibration"].column_names,
    )
    recipe = get_quantization_recipe(precision, args.smoothing_strength)

    print(f"Starting {precision} quantization process...")
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
    print(f"Saving quantized model to: {save_dir}...")
    save_model(model, processor, save_dir)
    config_path = save_dir / "config.json"
    print(f"Postprocessing config file {config_path}...")
    postprocess_config(config_path)
    print(f"Quantization complete! Model saved to: {save_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
