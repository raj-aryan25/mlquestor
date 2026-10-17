.PHONY: test lint typecheck build

test:
	uv run --locked --extra dev python -m pytest

lint:
	uv run --locked --extra dev python -m ruff check src tests

typecheck:
	uv run --locked --extra dev python -m mypy src/mlquest

build:
	uv run --locked --extra dev python -m build
