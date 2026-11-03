.PHONY: install format lint test check clean

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check .

fix:
	uv run ruff check . --fix

test:
	uv run pytest -v

check: format lint test

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +