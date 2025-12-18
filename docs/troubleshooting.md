# Troubleshooting

## Resources

* [vLLM Troubleshooting](https://docs.vllm.ai/en/latest/usage/troubleshooting/#hangs-loading-a-model-from-disk)

## Common Fixes

1. Re-install: `uv sync --extra cu128 --reinstall && source .venv/bin/activate`

## Common Errors

### OpenAI API Connection Error

Error message: `openai.APIConnectionError: Connection error.`

Check the server log. Common issues:

1. Server is not fully started. Wait until you see `Application startup complete.`.
1. Server died due to Out of Memory (OOM).
    1. Verify your GPU satisfies the [minimum requirements](../README.md#inference).
    1. Reduce `--max-model-len`. Recommended range: 8192 - 16384.

### PTXAS Error

Error message: `(EngineCore_DP0 pid=1477831) ptxas fatal   : Value 'sm_121a' is not defined for option 'gpu-name'`

Fix: Use CUDA 13.0 Docker container.
