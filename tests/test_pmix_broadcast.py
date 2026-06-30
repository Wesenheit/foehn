import os
import subprocess
import sys

import pytest

import rixa


def is_worker():
    return os.environ.get("RIXA_WORKER_MODE") == "1"


if is_worker():
    store = rixa.PMIxStore(30)
    root = 0
    key = "test"
    rank = store.get_rank()
    if rank == root:
        val = b"test_value"
    else:
        val = None
    out = store.broadcast(key, val, root)
    assert out == b"test_value"
    print("RIXA_CPU_WORKER_CLEAN_EXIT")
    sys.exit(0)


@pytest.mark.cpu
def test_pmix_broadcast(nprocs):
    env = os.environ.copy()
    env["RIXA_WORKER_MODE"] = "1"

    cmd = [
        "mpirun",
        "-n",
        nprocs,
        "-x",
        "RIXA_WORKER_MODE",
        "-x",
        "PYTHONPATH",
        sys.executable,
        __file__,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert result.stdout.count("RIXA_CPU_WORKER_CLEAN_EXIT") == int(nprocs)
