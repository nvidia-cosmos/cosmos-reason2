<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Paper](https://arxiv.org/abs/2503.15558) [HuggingFace](https://huggingface.co/collections/nvidia/cosmos-reason2) | [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)

NVIDIA Cosmos Reason – an open, customizable, 7B-parameter reasoning vision language model (VLM) for physical AI and robotics - enables robots and vision AI agents to reason like humans, using prior knowledge, physics understanding and common sense to understand and act in the real world. This model understands space, time, and fundamental physics, and can serve as a planning model to reason what steps an embodied agent might take next.
Cosmos Reason excels at navigating the long tail of diverse scenarios of the physical world with spatial-temporal understanding. Cosmos Reason is post-trained with physical common sense and embodied reasoning data with supervised fine-tuning and reinforcement learning. It uses chain-of-thought reasoning capabilities to understand world dynamics without human annotations.

## News

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
  uv tool install -U "huggingface_hub[cli]"
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

Cosmos-Reason2 is included in [`transformers>=4.57.0`](https://huggingface.co/docs/transformers/en/index).

We provide example inference scripts:

* [Minimal example](scripts/inference_sample.py)

  ```shell
  uv run scripts/inference_sample.py
  ```

* [Full example](scripts/inference.py)

  Caption the video:

  ```shell
  uv run scripts/inference.py --prompt prompts/caption.yaml --videos assets/sample.mp4 -v
  ```

  Ask a question about the video with reasoning:

  ```shell
  uv run scripts/inference.py --prompt prompts/question.yaml --question 'What are the potential safety hazards?' --reasoning --videos assets/sample.mp4 -v
  ```

  Temporally caption the video and save the input frames to `outputs/temporal_caption_text` for debugging:

  ```shell
  uv run scripts/inference.py --prompt prompts/temporal_caption_text.yaml --videos assets/sample.mp4 --timestamp -v -o outputs/temporal_caption_text
  ```

  Configure inference by editing:

  * [Prompts](prompts/README.md)
  * [Sampling Parameters](configs/sampling_params.yaml)
  * [Vision Processor Config](configs/vision_config.yaml)

## Tutorials

## Post-Training

The [nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) repository is an async post-training framework specialized for Supervised Fine-Tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). It prioritizes performance, scalability, and fault tolerance.

## Additional Resources

The Cosmos-Reason2 model is based on the Qwen3-VL model architecture. Useful resources:

* [Repository](https://github.com/QwenLM/Qwen3-VL)
* [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
