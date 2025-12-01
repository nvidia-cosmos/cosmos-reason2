default:
  just --list

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
