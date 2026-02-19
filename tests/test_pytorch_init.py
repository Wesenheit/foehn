import os
import sys
import subprocess
import pytest
import torch
import foehn

WORKER_ENV_VAR = "FOEHN_PYTORCH_WORKER"


def is_worker():
    return os.environ.get(WORKER_ENV_VAR) == "1"


if is_worker():
    try:
        foehn.pytorch.init_process_group()

        is_init = torch.distributed.is_initialized()

        if is_init:
            torch.distributed.destroy_process_group()
            print("PYTORCH_WORKER_CLEAN_EXIT")

        sys.exit(0 if is_init else 1)
    except Exception as e:
        print(f"WORKER_ERROR: {e}")
        sys.exit(1)


def test_pytorch_init_isolated(nprocs):
    env = os.environ.copy()
    env[WORKER_ENV_VAR] = "1"

    cmd = [
        "prterun",
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
    assert result.stdout.count("PYTORCH_WORKER_CLEAN_EXIT") == 2


if __name__ == "__main__":
    if not is_worker():
        sys.exit(pytest.main([__file__]))
