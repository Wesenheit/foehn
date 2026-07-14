import numpy as np
import nvshmem.core as nvshmem
from cuda.core import Device

from rixa.PMIx_core import PMIxStore


def init(dev: Device, store: PMIxStore) -> None:

    rank = store.get_rank()
    uid = nvshmem.get_unique_id(empty=(rank != 0))
    uid_bytes = uid._data.view(np.uint8).tobytes()
    uid_bytes = store.broadcast("uid", uid_bytes, 0)
    uid._data.view(np.uint8)[:] = np.frombuffer(uid_bytes, dtype=np.uint8)

    nvshmem.init(
        device=dev,
        uid=uid,
        rank=rank,
        nranks=store.get_world(),
        initializer_method="uid",
    )
