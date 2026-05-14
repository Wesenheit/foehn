from datetime import timedelta
from foehn.PMIx_core import PMIxStore
import torch.distributed as dist
from typing import overload


class FoehnPMIxStore(dist.Store):
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


def init_process_group(*args, **kwargs):
    store = FoehnPMIxStore()
    rank = store._store.get_rank()
    world = store._store.get_world()
    dist.init_process_group(*args, **kwargs, store=store, rank=rank, world_size=world)
