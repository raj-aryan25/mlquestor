.PHONY: test lint typecheck build

test:
	python -m pytest

lint:
	python -m ruff check src tests

typecheck:
	python -m mypy src/mlquest

build:
	python -m build
