import os
import subprocess
import sys

import pytest

WORKER_ENV_VAR = "RIXA_NVSHMEM_WORKER"


def is_worker():
    return os.environ.get(WORKER_ENV_VAR) == "1"


if is_worker():
    try:
        import nvshmem.core as nvshmem
        from cuda.core import Device

        import rixa.nvshmem

        store = rixa.PMIxStore(30)
        dev = Device(0)
        rixa.nvshmem.init(dev, store)

        is_init = nvshmem.init_status()
        if is_init:
            nvshmem.finalize()
            print("NVSHMEM_WORKER_CLEAN_EXIT")

        sys.exit(0 if is_init else 1)
    except Exception as e:
        print(f"WORKER_ERROR: {e}")
        sys.exit(1)


@pytest.mark.gpu
def test_nvshmem_init_isolated(nprocs_gpu):
    env = os.environ.copy()
    env[WORKER_ENV_VAR] = "1"

    cmd = [
        "mpirun",
        "-n",
        nprocs_gpu,
        "-x",
        WORKER_ENV_VAR,
        "-x",
        "PYTHONPATH",
        sys.executable,
        __file__,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    assert result.returncode == 0, f"Subprocess failed with stderr: {result.stderr}"
    assert result.stdout.count("NVSHMEM_WORKER_CLEAN_EXIT") == int(nprocs_gpu)


if __name__ == "__main__":
    if not is_worker():
        sys.exit(pytest.main([__file__]))
