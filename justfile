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
lint: _pre-commit-install _pre-commit-base _pre-commit

# Run pyrefly
_pyrefly *args:
  uv run pyrefly check --output-format=min-text --remove-unused-ignores {{args}}

_pyrefly-ignore *args: (_pyrefly '--suppress-errors' args)

# Run tests
test:
  uv run pytest

# Run pip-licenses
_pip-licenses *args: install
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md {{args}}
  pre-commit run --files ATTRIBUTIONS.md || true

# Update the license
license: _pip-licenses

docker_build_args := ''
docker_run_args := '--ipc=host -v /root/.cache:/root/.cache'

# Run the docker container
_docker build_args='' run_args='':
  #!/usr/bin/env bash
  set -euxo pipefail
  docker build {{docker_build_args}} {{build_args}} .
  image_tag=$(docker build {{docker_build_args}} {{build_args}} -q .)
  docker run \
    -it \
    --gpus all \
    --rm \
    -v .:/workspace \
    {{docker_run_args}} \
    {{run_args}} \
    $image_tag

docker-cu130 *run_args: (_docker '-f docker/nightly.Dockerfile' run_args)
