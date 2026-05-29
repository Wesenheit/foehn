from datetime import timedelta
import torch.distributed as dist
from rixa.PMIx_core import PMIxStore
from typing import overload
from rixa._rixa_torch import PMIxC10dStore


class PyTorchPMIxStore(dist.Store):
    def __init__(self, timeout: int = 30):
        super().__init__()
        self._store = PMIxStore(timeout)

    def set(self, key, value):
        self._store.set(key, value)

    def get(self, key):
        return self._store.get(key)

    @overload
    def wait(self, keys: list[str]) -> None: ...

    @overload
    def wait(self, keys: list[str], timeout: timedelta) -> None: ...

    def wait(self, keys: list[str], timeout: timedelta | None) -> None:
        if timeout is not None:
            timeout_seconds = int(timeout.total_seconds())
        else:
            timeout_seconds = -1
        return self._store.wait(keys, timeout_seconds)


def init_process_group(backend, *args, **kwargs):
    store = PMIxC10dStore()
    rank = store.rank()
    world = store.world_size()
    dist.init_process_group(*args, **kwargs, store=store, rank=rank, world_size=world)
