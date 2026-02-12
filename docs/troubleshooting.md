# Troubleshooting

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Resources](#resources)
- [FAQ](#faq)
  - [Where is requirements.txt](#where-is-requirementstxt)
- [Migration](#migration)
  - [Reason 1 → Reason 2 Configuration Changes](#reason-1--reason-2-configuration-changes)
  - [System Prompt (from Cosmos Cookbook)](#system-prompt-from-cosmos-cookbook)
- [Errors](#errors)
  - [EngineCore Issues](#enginecore-issues)
  - [OpenAI API Connection Error](#openai-api-connection-error)
  - [PTXAS Error](#ptxas-error)

______________________________________________________________________

<!--TOC-->

## Resources

* [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/#hangs-loading-a-model-from-disk)

## FAQ

### Where is requirements.txt

For most use cases, you should not need `requirements.txt`. `pip` can install directly from `pyproject.toml`. See the [nightly Dockerfile](../docker/nightly.Dockerfile) for an example installing into the NVIDIA vLLM container.

You can generate a `requirements.txt` file with [`uv export`](https://docs.astral.sh/uv/concepts/projects/export/).

```shell
uv export --format requirements.txt --output-file requirements.txt
```

## Migration

### Reason 1 → Reason 2 Configuration Changes

If you have custom inference scripts or config files from Cosmos Reason 1, update them to use the updated format:

**Reason 2 format:**

```json
"mm_processor_kwargs": {
  "size": {
    "shortest_edge": 1568,
    "longest_edge": 374544
  }
}
```

**Reason 1 format (deprecated):**

```json
"mm_processor_kwargs": {
  "videos_kwargs": {
    "min_pixels": 1568,
    "max_pixels": 374544
  }
}
```

**What changed:**

- **Vision processor arguments**: Updated to follow [Qwen3-VL Pixel Control specification](https://github.com/QwenLM/Qwen3-VL#:~:text=Pixel%20Control%20via%20Official%20Processor)
  - **Important**: The semantics of Qwen2.5 vs Qwen3 preprocessing parameters are different. You **cannot** simply replace `min_pixels`/`max_pixels` with `shortest_edge`/`longest_edge` using the same values.
  - The `shortest_edge` and `longest_edge` parameters control resizing differently than `min_pixels` and `max_pixels`.
  - **Action required**: Recalculate appropriate values for `shortest_edge` and `longest_edge` based on your desired frame sizes. Refer to the [Qwen3-VL documentation](https://github.com/QwenLM/Qwen3-VL#:~:text=Pixel%20Control%20via%20Official%20Processor) for guidance on choosing appropriate values.
- **Video timestamps**: No longer overlay timestamps on videos. Timestamps are now included in the model embedding automatically.

### System Prompt (from [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/prompt_guide/reason_guide.html#system-prompt))

In Cosmos Reason 2, we are more aligned with Qwen's use of system prompt. In the examples given, we simply use 'You are a helpful assistant.' You may note this is different from Reason 1, where the system prompt was heavily used.

## Errors

### EngineCore Issues

**Error message**: `ERROR EngineCore failed to start` or `RuntimeError: Engine core initialization failed`

This typically indicates GPU resource issues.

**1. Check GPU Memory:**

```bash
nvidia-smi
```

Look for:

- Available VRAM vs. model requirements
- Other processes using GPU memory

**2. Kill Zombie Processes:**

```bash
# Find processes using GPU
nvidia-smi

# Kill stuck processes (use PID from nvidia-smi output)
kill -9 <PID>
```

**Common causes:**

- Insufficient VRAM (model requires more than available)
- Another process holding GPU memory
- Driver/runtime version mismatch
- Corrupted CUDA cache (try: `rm -rf ~/.cache/huggingface/`)

### OpenAI API Connection Error

Error message: `openai.APIConnectionError: Connection error.`

Check the server log. Common issues:

1. Server is not fully started. Wait until you see `Application startup complete.`.
1. Server died due to Out of Memory (OOM).
    1. Verify your GPU satisfies the [minimum requirements](../README.md#inference).
    1. Reduce `--max-model-len`. Recommended range: 8192 - 16384.

### PTXAS Error

Error message: `(EngineCore_DP0 pid=1477831) ptxas fatal   : Value 'sm_121a' is not defined for option 'gpu-name'`

Fix: Use CUDA 13.0 [Docker container](../README.md#setup)

Alternatively, to use the virtual environment, set `TRITON_PTXAS_PATH` to your system `PTXAS`:

```shell
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
```

Your system CUDA version must match the torch CUDA version.
