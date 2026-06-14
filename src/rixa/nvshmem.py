import numpy as np
import nvshmem.core as nvshmem
from cuda.core import Device

from rixa.PMIx_core import PMIxStore


def init(dev: Device, store: PMIxStore, timeout: int = 30) -> None:

    rank = store.get_rank()
    uid = nvshmem.get_unique_id(empty=(rank != 0))
    uid_bytes = uid._data.view(np.uint8).tobytes()
    if rank == 0:
        store.set("root_uid", uid_bytes)

    store.wait(
        [
            "root_uid",
        ]
    )

    uid_bytes = store.get("root_uid")
    uid._data.view(np.uint8)[:] = np.frombuffer(uid_bytes, dtype=np.uint8)

    nvshmem.init(
        uid=uid,
        rank=rank,
        nranks=store.get_world(),
        initializer_method="uid",
    )
