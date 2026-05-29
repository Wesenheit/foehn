from datetime import timedelta
import torch.distributed as dist
from rixa.PMIx_core import PMIxStore
from typing import overload
import os
import importlib.util
from pathlib import Path
import ctypes

_ext = None


def load_pmix():
    libdir = Path(__file__).resolve().parent.parent / "rixa.libs"

    candidates = list(libdir.glob("libpmix*.so*"))
    if not candidates:
        raise RuntimeError("PMIx not found")
    lib = sorted(candidates)[0]  # or better selection logic
    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext  # already loaded this process

    if importlib.util.find_spec("rixa._rixa_torch"):
        import rixa._rixa_torch as m

        _ext = m
        return _ext

    try:
        import torch.utils.cpp_extension as cpp_ext
    except ImportError:
        raise RuntimeError(
            "PyTorch is required for this feature.\n"
            "Install it with: pip install rixa[torch]"
        )

    src_dir = os.path.join(os.path.dirname(__file__), "bindings")
    load_pmix()

    _ext = cpp_ext.load(
        name="_rixa_torch",  # no dots in JIT name
        sources=[
            os.path.join(src_dir, "rixa_pmix_store.c"),
            os.path.join(src_dir, "rixa_C10.cpp"),
        ],
        extra_cflags=[
            "-O3",
        ],
        extra_ldflags=["-Wl,-rpath,$ORIGIN/../rixa.libs"],
        # extra_ldflags=["-lpmix"],
        extra_include_paths=[
            src_dir,
        ],
        verbose=False,
    )
    return _ext


def get_pmix_store():
    return _get_ext().PMIxC10dStore


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
    store = get_pmix_store()()
    rank = store.rank()
    world = store.world_size()
    dist.init_process_group(
        backend, *args, **kwargs, store=store, rank=rank, world_size=world
    )
