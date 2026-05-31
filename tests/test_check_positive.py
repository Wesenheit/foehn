import sys
import os
import subprocess
from rixa.pytorch import get_pmix_store


def is_worker():
    return os.environ.get("RIXA_WORKER_MODE") == "1"


if is_worker():
    store = get_pmix_store()()
    rank = store.rank()
    if rank == 0:
        store.set("test_values", "1")

    assert store.check(["test_values"])

    print(f"WORKER_SUCCESS_RANK_{rank}")
    sys.exit(0)


def test_pmix_rank_validity(nprocs):
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
