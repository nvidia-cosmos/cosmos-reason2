# Troubleshooting

## Resources

* [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/#hangs-loading-a-model-from-disk)

## Common Fixes

1. Re-install: `uv sync --extra cu128 --reinstall && source .venv/bin/activate`

## Common Errors

### vLLM OOM

Reduce model length: `--max-model-len 16384`

### vLLM Memory Utilization

Error message: `Free memory on device (110.49/122.82 GiB) on startup is less than desired GPU memory utilization (0.9, 110.54 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.`

Limit vLLM memory utilization: `--gpu-memory-utilization 0.7`

### OpenAI API Connection Error

Error message: `openai.APIConnectionError: Connection error.`

Check the server log. Common issues:

1. Server is not fully started. Wait until you see `Application startup complete.`.
1. Server died due to Out of Memory (OOM).
    1. Verify your GPU satisfies the [minimum requirements](../README.md#inference).
    1. See [vLLM OOM](#vllm-oom) and [vLLM Memory Utilization](#vllm-memory-utilization).

### PTXAS Error

Error message: `(EngineCore_DP0 pid=1477831) ptxas fatal   : Value 'sm_121a' is not defined for option 'gpu-name'`

Fix: Use CUDA 13.0 Docker container.
