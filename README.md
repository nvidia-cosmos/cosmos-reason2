<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### [Paper](https://arxiv.org/abs/2503.15558) [HuggingFace](https://huggingface.co/collections/nvidia/cosmos-reason2) | [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)

NVIDIA Cosmos Reason – an open, customizable, reasoning vision language model (VLM) for physical AI and robotics - enables robots and vision AI agents to reason like humans, using prior knowledge, physics understanding and common sense to understand and act in the real world. This model understands space, time, and fundamental physics, and can serve as a planning model to reason what steps an embodied agent might take next.
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

### Transformers

Cosmos-Reason2 is included in [`transformers>=4.57.0`](https://huggingface.co/docs/transformers/en/index).

[Minimal example](scripts/inference_sample.py):

```shell
uv run scripts/inference_sample.py
```

### Deployment

For deployment and batch inference, we recommend using [`vllm`](https://docs.vllm.ai/en/stable/).

* [Online Serving](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
* [Offline Inference](https://docs.vllm.ai/en/stable/serving/offline_inference/)
* [Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)
* [LoRA](https://docs.vllm.ai/en/stable/features/lora/)

#### Online Serving

Start the server:

```shell
uv run vllm serve nvidia/Cosmos-Reason2-2B \
  --max-model-len 8192 \
  --allowed-local-media-path "$(pwd)" \
  --media-io-kwargs '{"video": {"num_frames": -1}}'
```

Caption the video:

```shell
uv run cosmos-reason2 online --prompt prompts/caption.yaml --videos assets/sample.mp4 --fps 2 -v
```

Embodied reasoning:

```shell
uv run cosmos-reason2 online --prompt prompts/embodied_reasoning.yaml --reasoning --images assets/sample.png
```

To list available parameters:

```shell
uv run cosmos-reason2 online --help
```

#### Offline Inference

Temporally caption the video and save the input frames to `outputs/temporal_caption_text` for debugging:

```shell
uv run cosmos-reason2 offline --prompt prompts/temporal_localization.yaml --videos assets/sample.mp4 --fps 2 -v -o outputs/temporal_localization
```

To list available parameters:

```shell
uv run cosmos-reason2 offline --help
```

## Quantization

For model quantization, we recommend using [llmcompressor](https://github.com/vllm-project/llm-compressor)

[Example](scripts/quantize.py):

```shell
./scripts/quantize.py
```

## Tutorials

* [Post-Training](examples/post_training/README.md)
  * [Hugging Face example](examples/post_training_hf/README.md)
  * [Llava example](examples/post_training_llava/README.md)

## Additional Resources

The Cosmos-Reason2 model is based on the Qwen3-VL model architecture. Useful resources:

* [Repository](https://github.com/QwenLM/Qwen3-VL)
* [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
