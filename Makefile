.PHONY: sync
sync:
	uv sync --all-extras --all-packages --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run ty check
	uv run pyright

.PHONY: check
check: format type

.PHONY: test
test:
	env -u PYTHONPATH uv run pytest

.PHONY: test-fast
test-fast:
	env -u PYTHONPATH uv run pytest -m "not slow"

.PHONY: test-cpu
test-cpu:
	FORCE_CPU=1 env -u PYTHONPATH uv run pytest

.PHONY: test-cpu-fast
test-cpu-fast:
	FORCE_CPU=1 env -u PYTHONPATH uv run pytest -m "not slow"

.PHONY: test-all
test-all: check test
