import pytest
from foehn.PMIx_core import PMIxStore
import sys
import os
import subprocess


def is_worker():
    return os.environ.get("FOEHN_WORKER_MODE") == "1"


if is_worker():
    store = PMIxStore(30)
    rank = store.get_rank()
    world = store.get_world()

    assert rank >= 0
    assert rank < world
    print(f"WORKER_SUCCESS_RANK_{rank}")
    sys.exit(0)


def test_pmix_rank_validity(nprocs):
    env = os.environ.copy()
    env["FOEHN_WORKER_MODE"] = "1"

    cmd = [
        "prterun",
        "-n",
        nprocs,
        "-x",
        "FOEHN_WORKER_MODE",
        "-x",
        "PYTHONPATH",
        sys.executable,
        __file__,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert "WORKER_SUCCESS_RANK_0" in result.stdout
    assert "WORKER_SUCCESS_RANK_1" in result.stdout
