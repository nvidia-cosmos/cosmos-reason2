# Post-Training using Cosmos-RL

[cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) is an async post-training framework specialized for Supervised Fine-Tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). It prioritizes performance, scalability, and fault tolerance.

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Setup](#setup)
  - [Virtual Environment](#virtual-environment)
  - [Docker Container](#docker-container)
  - [Install](#install)
  - [Monitor](#monitor)
- [Training](#training)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [Reinforcement Learning (RL)](#reinforcement-learning-rl)
- [Troubleshooting](#troubleshooting)
  - [Where are worker logs stored?](#where-are-worker-logs-stored)
  - [Where are checkpoints stored?](#where-are-checkpoints-stored)

______________________________________________________________________

<!--TOC-->

## Setup

### Virtual Environment

Prerequisites:

- [Setup](../../README.md#virtual-environment)

Install system dependencies:

- [redis](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/)

```shell
conda install -c conda-forge redis-server
```

### Docker Container

Prerequisites:

- [Setup](../../README.md#docker-container)

### Install

Install the example:

```shell
cd examples/cosmos_rl
uv sync
```

### Monitor

[Optional] We recommend that you to use [wandb](https://wandb.ai/) for training monitoring.

- Acquire your [WANDB_API_KEY](https://wandb.ai/authorize).
- Login:

```bash
uv tool install -U wandb
wandb login
```

When you run training, you will see the `wandb` link in the controller log:

```bash
wandb: 🚀 View run at https://wandb.ai/${WANDB_USER_NAME}/${config.logging.project_name}/runs/20250515101157
```

## Training

cosmos-rl is configured via a TOML file ([full config](../../configs/cosmos_rl_config.toml)).

We provide a post-training example using the [LLaVA datasets](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md) format.

Please update the fields `annotation_path` and `media_path` in the [config](configs/llava_sft.toml) to your custom dataset. `media_path` can be left as empty (`""`) if the paths in your annotation are absolute paths.

Here is one example of downloading the [Llava-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) dataset and [COCO](https://cocodataset.org/#home) images:

```shell
DATASET_DIR="/tmp/cosmos_reason2/cosmos_rl/data/llava_sft"
mkdir -p $DATASET_DIR
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json -O $DATASET_DIR/annotations.json
wget http://images.cocodataset.org/zips/train2017.zip -O $DATASET_DIR/media.zip && unzip -q $DATASET_DIR/media.zip -d $DATASET_DIR
```

### Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning (SFT) training can improve the model's capability on certain tasks with a similar distribution of the training dataset.

Minimum requirements:

- 4 GPUs with 80GB of memory

[Config](configs/llava_sft.toml) variants:

- 4 GPUs

  ```toml
  [policy.parallelism]
  dp_shard_size = 4
  ```

- 8 GPUs

  ```toml
  [policy.parallelism]
  dp_shard_size = 8
  ```

Run SFT ([sample outputs](../../assets/outputs/cosmos_rl_llava_sft)):

```shell
uv run cosmos-rl --config configs/llava_sft.toml --log-dir outputs/llava_sft scripts/llava_sft.py
```

### Reinforcement Learning (RL)

Coming soon!

## Troubleshooting

### Where are worker logs stored?

The worker log directory can be found in the console log:

```log
INFO:cosmos:Output files: ['outputs/llava_sft/logs_20251209-004947/policy_0.log']
```

The log directory contains the following:

* `controller.log`: Controller log: wandb info
* `policy_0.log`: Policy log: convergence progress

### Where are checkpoints stored?

After training finishes, the output checkpoint directory can be found in the policy log:

```log
[rank1]:[cosmos] 2025-12-06 02:04:31,300 - cosmos - INFO - [Policy] Step: 1, checkpoint saved successfully at /tmp/cosmos-reason2/post_training_hf/sft/20251206020308/checkpoints/step_1/policy.
```
