num_procs := "2"
export PYTHONPATH := "build/src"
    
setup:
    uvx meson setup build

build:
    uvx meson compile -C build

test: build
    uv pip install -e . --no-build-isolation
    uv run pytest
