import os
import subprocess
import sys

import jax
import pytest

WORKER_ENV_VAR = "RIXA_JAX_WORKER"


def is_worker():
    return os.environ.get(WORKER_ENV_VAR) == "1"


if is_worker():
    try:
        import rixa.jax

        store = rixa.PMIxStore(30)
        rixa.jax.init(store)

        is_init = jax.distributed.is_initialized()
        if is_init:
            print("JAX_WORKER_CLEAN_EXIT")

        sys.exit(0 if is_init else 1)
    except Exception as e:
        print(f"WORKER_ERROR: {e}")
        sys.exit(1)


@pytest.mark.cpu
def test_jax_init_isolated(nprocs):
    env = os.environ.copy()
    env[WORKER_ENV_VAR] = "1"

    cmd = [
        "mpirun",
        "-n",
        nprocs,
        "-x",
        WORKER_ENV_VAR,
        "-x",
        "PYTHONPATH",
        sys.executable,
        __file__,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    assert result.returncode == 0, f"Subprocess failed with stderr: {result.stderr}"
    assert result.stdout.count("JAX_WORKER_CLEAN_EXIT") == int(nprocs)


if __name__ == "__main__":
    if not is_worker():
        sys.exit(pytest.main([__file__]))
