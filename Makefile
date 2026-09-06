.PHONY: install test lint
install:
	uv sync

test:
	uv run pytest -v

lint:
	uv run ruff check src api tests