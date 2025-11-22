default:
  just --list

# Setup the repository
_pre-commit-install:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config-base.yaml

# Run linting and formatting
lint: _pre-commit-install
  pre-commit run --all-files || pre-commit run --all-files

# Run pyrefly
_pyrefly *args:
  uv run pyrefly check --output-format=min-text --remove-unused-ignores {{args}}

_pyrefly-ignore *args: (_pyrefly '--suppress-errors' args)

# Run tests
test: _pyrefly-ignore
  uv run pytest
