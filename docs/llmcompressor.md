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

Common arguments:

* `--model nvidia/Cosmos-Reason2-2B`: Model name or path.
* `--precision fp4`: Precision to use for quantization.
