# Notebooks

This directory contains a collection of Jupyter notebooks.

| Notebook | Description | Open in Colab |
| --- | --- | --- |
| [`trl_sft.ipynb`](./trl_sft.ipynb) | Supervised Fine-Tuning (SFT) with QLoRA using TRL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nvidia-cosmos/cosmos-reason2/blob/main/examples/notebooks/trl_sft.ipynb) |
| [`trl_grpo.ipynb`](./trl_grpo.ipynb) | GRPO with QLoRA using TRL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nvidia-cosmos/cosmos-reason2/blob/main/examples/notebooks/trl_grpo.ipynb) |

## Run Locally

Prerequisites:

* [Setup](../../README.md#setup)

Install the package:

```shell
cd examples/notebooks
uv sync
source .venv/bin/activate
```

Notebooks can be run as regular Python scripts:

```shell
python <notebook>.py
```

Alternatively, they can be run as Jupyter notebooks using [VS Code](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code).

## Contributing

Notebooks are written in [jupytext py:percent](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format) format. To synchronize the `.py` and `.ipynb` files, run:

```shell
just notebooks-sync
```
