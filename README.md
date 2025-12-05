<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/collections/nvidia/cosmos-reason2">Hugging Face</a>&nbsp | <a href="https://github.com/nvidia-cosmos/cosmos-cookbook">Cosmos Cookbook</a>
</p>

NVIDIA Cosmos Reason – an open, customizable, reasoning vision language model (VLM) for physical AI and robotics - enables robots and vision AI agents to reason like humans, using prior knowledge, physics understanding and common sense to understand and act in the real world. This model understands space, time, and fundamental physics, and can serve as a planning model to reason what steps an embodied agent might take next.
Cosmos Reason excels at navigating the long tail of diverse scenarios of the physical world with spatial-temporal understanding. Cosmos Reason is post-trained with physical common sense and embodied reasoning data with supervised fine-tuning and reinforcement learning. It uses chain-of-thought reasoning capabilities to understand world dynamics without human annotations.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [News!](#news)
- [Setup](#setup)
- [Inference](#inference)
  - [Transformers](#transformers)
  - [Deployment](#deployment)
    - [Online Serving](#online-serving)
    - [Offline Inference](#offline-inference)
- [Quantization](#quantization)
- [Post-Training](#post-training)
- [Additional Resources](#additional-resources)
- [License and Contact](#license-and-contact)

______________________________________________________________________

<!--TOC-->

## News!

## Setup

> **This repository only contains documentation/examples/utilities. You do not need it to run inference. See [Inference example](scripts/inference_sample.py) for a minimal inference example. The following setup instructions are only needed to run the examples in this repository.**

Install system dependencies:

* [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

* [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

```shell
uv tool install -U huggingface_hub
hf auth login
```

Clone the repository:

```shell
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2
```

## Inference

Minimum Requirements:

* 1 GPU with 24GB memory

### Transformers

Cosmos-Reason2 is included in [`transformers>=4.57.0`](https://huggingface.co/docs/transformers/en/index).

[Minimal example](scripts/inference_sample.py):

```shell
uv run scripts/inference_sample.py
```

### Deployment

For deployment and batch inference, we recommend using [`vllm`](https://docs.vllm.ai/en/stable/).

#### Online Serving

Start the server:

```shell
uv run vllm serve nvidia/Cosmos-Reason2-2B \
  --allowed-local-media-path "$(pwd)" \
  --max-model-len 8192 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3
```

Explanation of arguments:

* `--max-model-len 8192`: Maximum model length to avoid OOM (~24 GB for 2B model).
* `--media-io-kwargs '{"video": {"num_frames": -1}}'`: Allow overriding FPS per sample.
* `--reasoning-parser qwen3`: Parse reasoning trace.

Wait a few minutes for the server to startup. Once complete, it will print `Application startup complete.`. Open a new terminal to run inference commands.

Caption the video ([sample output](assets/outputs/caption.txt)):

```shell
uv run cosmos-reason2-inference online -i prompts/caption.yaml --videos assets/sample.mp4 --fps 4
```

Embodied reasoning ([sample output](assets/outputs/embodied_reasoning.txt)) with verbose output:

```shell
uv run cosmos-reason2-inference online -v -i prompts/embodied_reasoning.yaml --reasoning --images assets/sample.png
```

To list available parameters:

```shell
uv run cosmos-reason2-inference online --help
```

#### Offline Inference

Temporally caption the video and save the input frames to `outputs/temporal_localization` for debugging ([sample output](assets/outputs/temporal_localization.txt)):

```shell
uv run cosmos-reason2-inference offline -v -i prompts/temporal_localization.yaml --videos assets/sample.mp4 --fps 4 -o outputs/temporal_localization
```

To list available parameters:

```shell
uv run cosmos-reason2-inference offline --help
```

## Quantization

For model quantization, we recommend using [llmcompressor](https://github.com/vllm-project/llm-compressor)

[Example](scripts/quantize.py):

```shell
./scripts/quantize.py
```

## Post-Training

* **cosmos-rl** Coming soon!

## Additional Resources

* Qwen3
  * [Qwen3-VL Repository](https://github.com/QwenLM/Qwen3-VL)
  * [Qwen3-VL vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
  * [Qwen3 Documentation](https://qwen.readthedocs.io/en/latest/)
* vLLM
  * [Online Serving](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
  * [Offline Inference](https://docs.vllm.ai/en/stable/serving/offline_inference/)
  * [Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
  * [LoRA](https://docs.vllm.ai/en/stable/features/lora/)

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
