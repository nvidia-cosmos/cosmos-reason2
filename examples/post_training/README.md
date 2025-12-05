# Cosmos-Reason2 Post-Training

The [nvidia-cosmos/cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) repository is an async post-training framework specialized for Supervised Fine-Tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). It prioritizes performance, scalability, and fault tolerance.

- [cosmos-rl documentation](https://nvidia-cosmos.github.io/cosmos-rl/).

## Setup

### Install

Prerequisites:

- [Setup](../../README.md#setup)

Install system dependencies:

- [redis](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/)

  ```shell
  conda install -c conda-forge redis-server
  ```

### Monitor

[Optional] We recommend that you to use [wandb](https://wandb.ai/) for training monitoring.

1. Acquire your [WANDB_API_KEY](https://wandb.ai/authorize).
1. Login:

  ```bash
  uv tool install -U wandb
  wandb login
  ```

When you run training, you will see the `wandb` link in the logging:

```bash
wandb: 🚀 View run at https://wandb.ai/${WANDB_USER_NAME}/${config.logging.project_name}/runs/20250515101157
```

## Training

cosmos-rl is configured via a TOML file. For documentation, refer to [cosmos_rl.policy.config.Config](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/policy/config/__init__.py).

### Supervised Fine-Tuning (SFT)

The SFT training can improve the model's capability on certain tasks with a similar distribution of the training dataset.

Minimum Requirements:

- 4 GPUs with 80GB of memory

Parallelism variants:

- 4 GPU

  ```toml
  [policy.parallelism]
  dp_shard_size = 4
  ```

- 8 GPU

  ```toml
  [policy.parallelism]
  dp_shard_size = 8
  ```

After training finishes, the final output checkpoint can be found in the log:

```log
[rank0]:Exported safetensors to ./outputs/sft/20250516061336/safetensors/final
```

### Reinforcement Learning (RL)

The RL training can improve the model's reasoning capability on certain tasks with the reasoning training dataset.

Minimum Requirements:

- 4 GPUs with 80GB of memory

Parallelism variants:

- 4 GPU

  ```toml
  [rollout.parallelism]
  tp_size = 2

  [policy.parallelism]
  dp_shard_size = 2
  ```

- 8 GPU

  ```toml
  [rollout.parallelism]
  tp_size = 4

  [policy.parallelism]
  dp_shard_size = 4
  ```

Similar to SFT training, the final output checkpoint can be found in the log.
