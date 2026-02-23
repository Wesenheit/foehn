build:
    uv sync

test *args: build
    uv pip install -e . --no-build-isolation
    uv run pytest {{args}}
