# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
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

# %% [markdown]
# # GRPO Cosmos-Reason2 with QLoRA using TRL
#
# **WORK IN PROGRESS!**
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nvidia-cosmos/cosmos-reason2/blob/main/examples/notebooks/trl_grpo.ipynb)
#
# - [TRL GitHub Repository](https://github.com/huggingface/trl)
# - [Official TRL Examples](https://huggingface.co/docs/trl/example_overview)
# - [Community Tutorials](https://huggingface.co/docs/trl/community_tutorials)

# %% [markdown]
# ## Install dependencies
#
# We'll install **TRL** with the **PEFT** extra, which ensures all main dependencies such as **Transformers** and **PEFT** (a package for parameter-efficient fine-tuning, e.g., LoRA/QLoRA) are included. Additionally, we'll install **bitsandbytes** to enable quantization of LLMs, reducing memory consumption for both inference and training.

# %%
# !pip install -Uq "trl[peft]==0.26.1" "bitsandbytes==0.49.0" "tensorboard==2.20.0" "math_verify==0.8.0"

# %% [markdown]
# ### Log in to Hugging Face
#
# Log in to your **Hugging Face** account to save your fine-tuned model, track your experiment results directly on the Hub or access gated models. You can find your **access token** on your [account settings page](https://huggingface.co/settings/tokens).

# %% tags=["active-ipynb"]
# from huggingface_hub import notebook_login
#
# notebook_login()

# %% [markdown]
# ## Load dataset
#
#
# We'll load the [**lmms-lab/multimodal-open-r1-8k-verified**](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) dataset from the Hugging Face Hub using the `datasets` library.
#
# This dataset contains maths problems with the image representing the problem,  along with the solution in thinking format specially tailored for VLMs. By training our model with this dataset, it'll improve its maths and thinking reasoning.
#

# %%
from datasets import load_dataset

dataset_id = "lmms-lab/multimodal-open-r1-8k-verified"
train_dataset = load_dataset(dataset_id, split="train[:5%]")

# %% [markdown]
# In addition to the `problem` and `image` columns, we also include a custom system prompt to tell the model how we'd like the generation.
#
# The system prompt is extracted from DeepSeek R1. Refer to [this previous recipe](https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl) for more details.
#
# We convert the dataset samples into conversation samples, including the system prompt and one image and problem description per sample, since this is how the GRPO trainer expects them.
#
# We also set `padding_side="left"` to ensure that generated completions during training are concatenated directly after the prompt, which is essential for GRPO to correctly compare token-level probabilities between preferred and rejected responses.

# %%
from transformers import AutoProcessor

model_name = "nvidia/Cosmos-Reason2-2B"
processor = AutoProcessor.from_pretrained(model_name, padding_side="left")

SYSTEM_PROMPT = """You are a helpful assistant.

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""


def make_conversation(example):
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": example["problem"]},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "image": example["image"],
    }


train_dataset = train_dataset.map(make_conversation)

# %% [markdown]
# Let's review one example to understand the internal structure:

# %%
train_dataset[0]

# %%
train_dataset = train_dataset.remove_columns(
    ["problem", "original_question", "original_answer"]
)

# %%
train_dataset[0]

# %% [markdown]
# ## Load model and configure LoRA/QLoRA
#
# This notebook can be used with two fine-tuning methods. By default, it is set up for **QLoRA**, which includes quantization using `BitsAndBytesConfig`. If you prefer to use standard **LoRA** without quantization, simply comment out the `BitsAndBytesConfig` configuration.

# %%
import torch
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
)

# %% [markdown]
# The following cell defines LoRA (or QLoRA if needed). When training with LoRA/QLoRA, we use a **base model** (the one selected above) and, instead of modifying its original weights, we fine-tune a **LoRA adapter** — a lightweight layer that enables efficient and memory-friendly training. The **`target_modules`** specify which parts of the model (e.g., attention or projection layers) will be adapted by LoRA during fine-tuning.

# %%
from peft import LoraConfig

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

# %% [markdown]
# ## Train model
#
# We'll configure **GRPO** using `GRPOConfig`, keeping the parameters minimal. You can adjust these settings if more resources are available. For full details on all available parameters, check the [TRL GRPOConfig documentation](https://huggingface.co/docs/trl/sft_trainer#trl.GRPOConfig).
#
# First, we need to define the rewards functions that the training algorithm will use to improve the model. In this case, we'll include two reward functions.
# We'll use a format reward that will reward the model when the output includes `<think>` and `<answer>` tags and additionally a length-based reward to discourage overthinking. Both functions have been extracted from [here](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py).

# %%
import re


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions
    ]
    return [1.0 if match else 0.0 for match in matches]


# %%
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def len_reward(completions, solution, **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = completions

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparsable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


# %% [markdown]
# After defining the reward function(s), we can define the `GRPOConfig`.

# %%
from trl import GRPOConfig

output_dir = "outputs/Cosmos-Reason2-2B-trl-grpo"

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    learning_rate=2e-5,
    # num_train_epochs=1,
    max_steps=100,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    # Parameters that control the data preprocessing
    per_device_train_batch_size=2,
    max_completion_length=1024,  # default: 256            # Max completion length produced during training
    num_generations=2,  # 2, # default: 8                  # Number of generations produced during training for comparison
    fp16=True,
    # Parameters related to reporting and saving
    output_dir=output_dir,  # Where to save model checkpoints and logs
    logging_steps=1,  # Log training metrics every N steps
    report_to="tensorboard",  # Experiment tracking tool
    # Hub integration
    push_to_hub=True,
    log_completions=True,
)

# %% [markdown]
# Configure the GRPO Trainer. We pass the previously configured `training_args`. We don't use eval dataset to maintain memory usage low but you can configure it.

# %%
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, len_reward],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)

# %% [markdown]
# Show memory stats before training

# %%
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %% [markdown]
# And train!

# %%
trainer_stats = trainer.train()

# %% [markdown]
# Show memory stats after training

# %%
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# %% [markdown]
# ## Saving fine tuned model
#
# In this step, we save the fine-tuned model both **locally** and to the **Hugging Face Hub** using the credentials from your account.

# %%
trainer.save_model(output_dir)
trainer.push_to_hub(dataset_name=dataset_id)

# %% [markdown]
# ## Load the fine-tuned model and run inference
#
# Now, let's test our fine-tuned model by loading the **LoRA/QLoRA adapter** and performing **inference**. We'll start by loading the **base model**, then attach the adapter to it, creating the final fine-tuned model ready for evaluation.

# %%
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

base_model = model_name
adapter_model = f"{output_dir}"  # Replace with your HF username or organization

model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model, dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_model)

processor = AutoProcessor.from_pretrained(base_model)

# %%
train_dataset[0]

# %%
from datasets import load_dataset

dataset_id = "lmms-lab/multimodal-open-r1-8k-verified"
train_dataset = load_dataset(dataset_id, split="train[:5%]")

problem = train_dataset[0]["problem"]
image = train_dataset[0]["image"]

messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": problem},
        ],
    },
]

# %%
messages

# %%
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
