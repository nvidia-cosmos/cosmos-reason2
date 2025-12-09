default:
  just --list

install *args:
  uv sync {{args}}

# Setup the repository
_pre-commit-install:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config-base.yaml

_pre-commit-base *args:
  pre-commit run -c .pre-commit-config-base.yaml -a {{args}}

_pre-commit *args:
  pre-commit run -a {{args}} || pre-commit run -a {{args}}

# Run linting and formatting
lint: _pre-commit-install _pre-commit-base _pre-commit license

# Run pyrefly
_pyrefly *args:
  uv run pyrefly check --output-format=min-text --remove-unused-ignores {{args}}

_pyrefly-ignore *args: (_pyrefly '--suppress-errors' args)

# Run tests
test:
  uv run pytest -vv

# Run pip-licenses
_pip-licenses *args:
  uv run --extra quantize pip-licenses --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md {{args}}
  pre-commit run --files ATTRIBUTIONS.md || true

# Update the license
license: _pip-licenses

# Export config defaults and schemas
export-configs *args:
  uv run --all-extras python scripts/export_configs.py {{args}}

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
    {{run_args}} \
    $image_tag

# Run the CUDA 12.8 docker container.
docker-cu128 *run_args: (_docker '' run_args)

# Run the CUDA 13.0 docker container.
docker-cu130 *run_args: (_docker '-f docker/nightly.Dockerfile' run_args)
