import sys
import os
import subprocess
from rixa.pytorch import get_pmix_store
import pytest


def is_worker():
    return os.environ.get("RIXA_WORKER_MODE") == "1"


if is_worker():
    store = get_pmix_store()()
    rank = store.rank()
    if rank == 0:
        store.set("test_value_0", "1")
    if rank == 1:
        store.set("test_value_1", "2")
    store.wait(["test_value_0", "test_value_1"])

    val1 = store.get("test_value_0")
    val2 = store.get("test_value_1")
    assert val1.decode("utf-8") == "1"
    assert val2.decode("utf-8") == "2"

    print(f"WORKER_SUCCESS_RANK_{rank}")
    sys.exit(0)


@pytest.mark.cpu
def test_pmix_set_get_wait(nprocs):
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
    assert "WORKER_SUCCESS_RANK_0" in result.stdout
    assert "WORKER_SUCCESS_RANK_1" in result.stdout
