# Troubleshooting

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Resources](#resources)
- [FAQ](#faq)
  - [Where is requirements.txt](#where-is-requirementstxt)
- [Errors](#errors)
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

## Errors

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
