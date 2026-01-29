# Quantization using llmcompressor

For model quantization, we recommend using [llmcompressor](https://github.com/vllm-project/llm-compressor).

> The follow examples should be run from the root of the repository.

[Example](../scripts/quantize.py) ([sample output](../assets/outputs/quantize.log)):

```shell
./scripts/quantize.py -o /tmp/cosmos-reason2/checkpoints
```

To list available arguments:

```shell
./scripts/quantize.py --help
```

## Quantization Options

| Option | Values | Default | Description |
| ------ | ------ | ------- | ----------- |
| `--model` | string | `nvidia/Cosmos-Reason2-2B` | Model name or local path |
| `--precision` | `nvfp4`, `fp8`, `fp8_dynamic` | `nvfp4` | `nvfp4` (smallest/fastest), `fp8` (better quality), `fp8_dynamic` (best quality, slower inference) |
| `--kv-precision` | `bf16`, `fp8` | `bf16` | KV cache precision |
| `--num-samples` | integer | `512` | Calibration samples. Increase for better accuracy & longer runtime |
| `--smoothing-strength` | 0.0-1.0 | `0.8` | SmoothQuant strength for handling outliers |
