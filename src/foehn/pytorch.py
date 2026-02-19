from foehn.PMIx_core import PMIxStore
import sys
import torch.distributed as dist


class FoehnPMIxStore(dist.Store):
    def __init__(self, timeout: int = 30):
        super().__init__()
        self._store = PMIxStore(timeout)

    def set(self, key, value):
        self._store.set(key, value)

    def get(self, key):
        return self._store.get(key)

    def wait(self, keys, timeout):
        timeout_seconds = int(timeout.total_seconds())
        return self._store.wait(keys, timeout_seconds)

    def add(self, key, amount):
        print(f"add {key} {amount}", file=sys.stderr)


def init_process_group(*args, **kwargs):
    store = FoehnPMIxStore()
    rank = store._store.get_rank()
    world = store._store.get_world()
    dist.init_process_group(*args, **kwargs, store=store, rank=rank, world_size=world)
