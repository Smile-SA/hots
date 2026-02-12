.PHONY: all sync install test lint format doc clean done

all: install done

# Sync a local virtual environment in .venv using uv + the lockfile.
sync:
	uv sync --extra dev --extra kafka

# Alias
install: sync

test:
	uv run -- pytest

lint:
	uv run -- ruff format --check .
	uv run -- ruff check hots tests

format:
	uv run -- ruff format .
	uv run -- ruff check --fix hots tests

doc:
	uv sync --extra doc
	uv run -- sphinx-build -b html doc/source doc/build

done:
	@echo "Installation finished successfully.\n\
	Run 'uv run -- hots path/to/config/file' to start the application or\n\
	'uv run -- hots --help' for command description"

clean:
	rm -rf .pytest_cache .eggs *.egg-info .ruff_cache
	find . -path ./.venv -prune -o -name "*.pyc" -o -name "*.pyo" -o -name "__pycache__" -print0 | xargs -r -0 rm -rv
	@echo "To remove the environment, delete ./.venv manually." >&2
