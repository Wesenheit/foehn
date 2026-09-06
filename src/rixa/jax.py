import socket
from typing import Optional

import jax

from rixa.PMIx_core import PMIxStore


def init(
    store: PMIxStore, override_coordinator_port: Optional[str] = None, root: int = 0
):

    rank = store.get_rank()
    world = store.get_world()
    if rank == root:
        hostname = socket.gethostname()

        if override_coordinator_port:
            port_id = override_coordinator_port
        else:
            # Original jax initalization from mpi4py code (https://github.com/jax-ml/jax/blob/main/jax/_src/clusters/mpi4py_cluster.py)
            port_id = str(hash(hostname) % 2**12 + (65535 - 2**12 + 1))

        hostname = f"{hostname}:{port_id}"

    else:
        hostname = "None"

    hostname_bytes = hostname.encode()
    hostname_bytes = store.broadcast("hostname", hostname_bytes, root)
    hostname = hostname_bytes.decode()
    jax.distributed.initialize(hostname, world, rank)
