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
- [Model Family](#model-family)
- [Setup](#setup)
- [Inference](#inference)
  - [Minimum GPU Memory](#minimum-gpu-memory)
  - [Tested Platforms](#tested-platforms)
  - [Transformers](#transformers)
  - [Deployment](#deployment)
    - [Online Serving](#online-serving)
    - [Offline Inference](#offline-inference)
- [Post-Training](#post-training)
- [Quantization](#quantization)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)
- [License and Contact](#license-and-contact)

______________________________________________________________________

<!--TOC-->

## News!

* [December 19, 2025] We have released the Cosmos-Reason2 models and code for Physical AI common sense and embodied reasoning. The 2B and 8B models are now available on Hugging Face.

## Model Family

* [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
* [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)

## Setup

> **This repository only contains documentation/examples/utilities. You do not need it to run inference. See [Inference example](scripts/inference_sample.py) for a minimal inference example. The following setup instructions are only needed to run the examples in this repository.**

Clone the repository:

```shell
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2
```

Install one of the following environments:

<details id="virtual-environment"><summary><b>Virtual Environment</b></summary>

Install system dependencies:

```shell
sudo apt-get install curl ffmpeg git git-lfs
```

* [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

* [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

```shell
uvx hf auth login
```

Install the repository:

```shell
uv sync --extra cu128
source .venv/bin/activate
```

CUDA variants:

| CUDA Version | Arguments | Notes |
| --- | --- | --- |
| CUDA 12.8 | `--extra cu128` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 13.0 | `--extra cu130` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

For DGX Spark and Jetson AGX, you must use CUDA 13.0. Additionally, you must set `TRITON_PTXAS_PATH` to your system `PTXAS`:

```shell
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
```

</details>

<details id="docker-container"><summary><b>Docker Container</b></summary>

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

Build the container:

```bash
image_tag=$(docker build -f Dockerfile --build-arg=CUDA_VERSION=12.8.1 -q .)
```

CUDA variants:

| CUDA Version | Arguments | Notes |
| --- | --- | --- |
| CUDA 12.8 | `--build-arg=CUDA_VERSION=12.8.1` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 13.0 | `--build-arg=CUDA_VERSION=13.0.0` | [NVIDIA Driver](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

For DGX Spark and Jetson AGX, you must use CUDA 13.0.

Run the container:

```bash
docker run -it --gpus all --ipc=host --rm -v .:/workspace -v /workspace/.venv -v /workspace/examples/cosmos_rl/.venv -v /root/.cache:/root/.cache -e HF_TOKEN="$HF_TOKEN" $image_tag
```

Optional arguments:

* `--ipc=host`: Use host system's shared memory, since parallel torchrun consumes a large amount of shared memory. If not allowed by security policy, increase `--shm-size` ([documentation](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)).
* `-v /root/.cache:/root/.cache`: Mount host cache to avoid re-downloading cache entries.
* `-e HF_TOKEN="$HF_TOKEN"`: Set Hugging Face token to avoid re-authenticating.

</details>

## Inference

### Minimum GPU Memory

| Model | GPU Memory |
| --- | --- |
| Cosmos-Reason2-2B | 24GB |
| Cosmos-Reason2-8B | 32GB |

### Tested Platforms

Cosmos-Reason2 works on Hopper and Blackwell. Additional hardware configurations may work but are not officially validated at the time of this release.

Examples have been tested on the following devices:

| GPU | CUDA Version | Functionality |
| --- | --- | --- |
| NVIDIA H100 | 12.8 | inference/post-training/quantization |
| NVIDIA GB200 | 13.0 | inference |
| NVIDIA DGX Spark | 13.0 | inference |
| NVIDIA Jetson AGX Thor (Edge) | 13.0 | Transformers inference. vLLM inference is coming soon! |

### Transformers

Cosmos-Reason2 is included in [`transformers>=4.57.0`](https://huggingface.co/docs/transformers/en/index).

[Minimal example](scripts/inference_sample.py) ([sample output](assets/outputs/sample.log)):

```shell
python scripts/inference_sample.py
```

### Deployment

For deployment and batch inference, we recommend using [`vllm>=0.11.0`](https://docs.vllm.ai/en/stable/).

#### Online Serving

Start the server in a separate terminal or a background process:

```shell
vllm serve nvidia/Cosmos-Reason2-2B \
  --allowed-local-media-path "$(pwd)" \
  --max-model-len 16384 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --port 8000
```

Optional arguments:

* `--max-model-len 16384`: Maximum model length to avoid OOM. Recommended range: 8192 - 16384.
* `--media-io-kwargs '{"video": {"num_frames": -1}}'`: Allow overriding FPS per sample.
* `--reasoning-parser qwen3`: Parse reasoning trace.
* `--port 8000`: Server port. Change if you encounter `Address already in use` errors.

Wait a few minutes for the server to startup. Once complete, it will print `Application startup complete.`.

Caption a video ([sample output](assets/outputs/caption.log)):

```shell
cosmos-reason2-inference online --port 8000 -i prompts/caption.yaml --reasoning --videos assets/sample.mp4 --fps 4
```

Embodied reasoning with verbose output ([sample output](assets/outputs/embodied_reasoning.log)):

```shell
cosmos-reason2-inference online -v --port 8000 -i prompts/embodied_reasoning.yaml --reasoning --images assets/sample.png
```

To list available arguments:

```shell
cosmos-reason2-inference online --help
```

#### Offline Inference

Temporally caption a video and save the input frames to `outputs/temporal_localization` for debugging ([sample output](assets/outputs/temporal_localization.log)):

```shell
cosmos-reason2-inference offline -v --max-model-len 16384 -i prompts/temporal_localization.yaml --videos assets/sample.mp4 --fps 4 -o outputs/temporal_localization
```

To list available arguments:

```shell
cosmos-reason2-inference offline --help
```

Common arguments:

* `--model nvidia/Cosmos-Reason2-2B`: Model name or path.

## Post-Training

* [TRL](examples/notebooks/README.md)
* [Cosmos-RL](examples/cosmos_rl/README.md)

## Quantization

* [llmcompressor](docs/llmcompressor.md)

## Troubleshooting

See [troubleshooting guide](docs/troubleshooting.md)

## Additional Resources

* [Troubleshooting](docs/troubleshooting.md)
* [Example prompts](prompts/README.md)
* Cosmos-Reason2 is based on the Qwen3-VL architecture.
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
