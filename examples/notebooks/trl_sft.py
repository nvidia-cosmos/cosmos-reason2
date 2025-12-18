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
# # Supervised Fine-Tuning (SFT) Cosmos-Reason2 with QLoRA using TRL
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nvidia-cosmos/cosmos-reason2/blob/main/examples/notebooks/trl_sft.ipynb)
#
# - [TRL GitHub Repository](https://github.com/huggingface/trl)
# - [Official TRL Examples](https://huggingface.co/docs/trl/example_overview)
# - [Community Tutorials](https://huggingface.co/docs/trl/community_tutorials)

# %% [markdown]
# ## Install dependencies
#
# We'll install **TRL** with the **PEFT** extra, which ensures all main dependencies such as **Transformers** and **PEFT** (a package for parameter-efficient fine-tuning, e.g., LoRA/QLoRA) are included. Additionally, we'll install **bitsandbytes** to enable quantization of LLMs, reducing memory consumption for both inference and training.

# %%
# !pip install -Uq "trl[peft]==0.26.1" "bitsandbytes==0.49.0" "tensorboard==2.20.0"

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
# We'll load the [**trl-lib/llava-instruct-mix**](https://huggingface.co/datasets/trl-lib/llava-instruct-mix) dataset from the Hugging Face Hub using the `datasets` library.
#
# This dataset is a set of GPT-generated multimodal instruction-following data. We use a processed version for conveniency here. You can check out more details about how to configure your own multimodal dataset for traininig with SFT in the [docs](https://huggingface.co/docs/trl/en/sft_trainer#training-vision-language-models). Fine-tuning Qwen3-VL on it helps refine its response style and visual understanding.
#

# %%
from datasets import load_dataset

dataset_name = "trl-lib/llava-instruct-mix"
train_dataset = load_dataset(dataset_name, split="train[:10%]")

# %% [markdown]
# Let's review one example to understand the internal structure:

# %%
train_dataset[0]

# %% [markdown]
# ## Load model and configure LoRA/QLoRA
#
# This notebook can be used with two fine-tuning methods. By default, it is set up for **QLoRA**, which includes quantization using `BitsAndBytesConfig`. If you prefer to use standard **LoRA** without quantization, simply comment out the `BitsAndBytesConfig` configuration.

# %%
import torch
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration

model_name = "nvidia/Cosmos-Reason2-2B"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,  # Load the model in 4-bit precision to save memory
        bnb_4bit_compute_dtype=torch.float16,  # Data type used for internal computations in quantization
        bnb_4bit_use_double_quant=True,  # Use double quantization to improve accuracy
        bnb_4bit_quant_type="nf4",  # Type of quantization. "nf4" is recommended for recent LLMs
    ),
)

# %% [markdown]
# The following cell defines LoRA (or QLoRA if needed). When training with LoRA/QLoRA, we use a **base model** (the one selected above) and, instead of modifying its original weights, we fine-tune a **LoRA adapter** — a lightweight layer that enables efficient and memory-friendly training. The **`target_modules`** specify which parts of the model (e.g., attention or projection layers) will be adapted by LoRA during fine-tuning.

# %%
from peft import LoraConfig

peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=[
        "down_proj",
        "o_proj",
        "k_proj",
        "q_proj",
        "gate_proj",
        "up_proj",
        "v_proj",
    ],
)

# %% [markdown]
# ## Train model
#
# We'll configure **SFT** using `SFTConfig`, keeping the parameters minimal. You can adjust these settings if more resources are available. For full details on all available parameters, check the [TRL SFTConfig documentation](https://huggingface.co/docs/trl/sft_trainer#trl.SFTConfig).

# %%
from trl import SFTConfig

output_dir = "outputs/Cosmos-Reason2-2B-trl-sft"

# Configure training arguments using SFTConfig
training_args = SFTConfig(
    # Training schedule / optimization
    # num_train_epochs=1,
    max_steps=10,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    per_device_train_batch_size=2,  # Batch size per GPU/CPU
    gradient_accumulation_steps=8,  # Gradients are accumulated over multiple steps → effective batch size = 4 * 8 = 32
    warmup_steps=5,  # Gradually increase LR during first N steps
    learning_rate=2e-4,  # Learning rate for the optimizer
    optim="adamw_8bit",  # Optimizer
    max_length=None,  # For VLMs, truncating may remove image tokens, leading to errors during training. max_length=None avoids it
    # Logging / reporting
    output_dir=output_dir,  # Where to save model checkpoints and logs
    logging_steps=1,  # Log training metrics every N steps
    report_to="tensorboard",  # Experiment tracking tool
)

# %% [markdown]
# Configure the SFT Trainer. We pass the previously configured `training_args`. We don't use eval dataset to maintain memory usage low but you can configure it.

# %%
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
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
# In this step, we save the fine-tuned model **locally**.

# %%
trainer.save_model(output_dir)

# %% [markdown]
# ## Load the fine-tuned model and run inference
#
# Now, let's test our fine-tuned model by loading the **LoRA/QLoRA adapter** and performing **inference**. We'll start by loading the **base model**, then attach the adapter to it, creating the final fine-tuned model ready for evaluation.

# %%
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

base_model = model_name
adapter_model = f"{output_dir}"  # Replace with your HF username or organization + fine-tuned model name

model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_model, dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_model)

processor = AutoProcessor.from_pretrained(base_model)

# %%
problem = train_dataset[0]["prompt"][0]["content"]
image = train_dataset[0]["images"][0]

messages = [
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
image

# %%
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

# %%
