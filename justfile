default:
  just --list

default_cuda_name := "cu128"

# Install the repository
install cuda_name=default_cuda_name *args:
  echo {{cuda_name}} > .cuda-name
  uv sync --extra={{cuda_name}} {{args}}

# Run uv sync
_uv-sync *args:
  if [ ! -f .cuda-name ]; then \
    echo {{default_cuda_name}} > .cuda-name; \
  fi
  uv sync --extra=$(cat .cuda-name) {{args}}

# Run a command in the package environment
run *args: _uv-sync
  uv run --no-sync {{args}}

# Setup the repository
_pre-commit-install:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config-base.yaml

_pre-commit-base *args:
  pre-commit run -c .pre-commit-config-base.yaml -a {{args}}

_pre-commit *args:
  pre-commit run -a {{args}} || pre-commit run -a {{args}}

# Run linting and formatting
lint: _pre-commit-install _pre-commit-base _pre-commit notebooks-sync license

# Run tests
test: _uv-sync
  uv run --no-sync pytest -vv

# Run pip-licenses
_pip-licenses *args: _uv-sync
  #!/usr/bin/env bash
  set -euxo pipefail
  temp_dir=$(mktemp -d)
  uv venv --clear $temp_dir
  python_path="$temp_dir/bin/python"
  uv pip install -r requirements.txt --python $python_path
  uv run --no-sync pip-licenses \
    --python $python_path \
    --format=plain-vertical \
    --with-license-file \
    --no-license-path \
    --no-version \
    --with-urls \
    --output-file ATTRIBUTIONS.md \
    --packages $(cat requirements.txt | cut -d '=' -f 1 | xargs) \
    {{args}}

# Update the license
license: _pip-licenses

# Sync jupytext notebooks
[working-directory: 'examples/notebooks']
notebooks-sync:
  uv run jupytext --sync *.ipynb

# Run the docker container
_docker build_args='' run_args='':
  #!/usr/bin/env bash
  set -euxo pipefail
  docker build {{build_args}} .
  image_tag=$(docker build {{build_args}} -q .)
  docker run \
    -it \
    --gpus all \
    --ipc=host \
    --rm \
    -v .:/workspace \
    -v /workspace/.venv \
    -v /workspace/examples/cosmos_rl/.venv \
    -v /root/.cache:/root/.cache \
    -e HF_TOKEN="$HF_TOKEN" \
    {{run_args}} \
    $image_tag

# Run the CUDA 12.8 docker container.
docker-cu128 *run_args: (_docker '--build-arg=CUDA_VERSION=12.8.1' run_args)

# Run the CUDA 13.0 docker container.
docker-cu130 *run_args: (_docker '--build-arg=CUDA_VERSION=13.0.0' run_args)
