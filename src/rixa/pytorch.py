import ctypes
import importlib.util
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils.cpp_extension as cpp_ext

_ext = None


def is_compiled_lazy():
    ext_path = cpp_ext._get_build_directory("_rixa_torch", False)
    return os.path.isdir(ext_path) and any(
        f.endswith(".so") for f in os.listdir(ext_path)
    )


def is_compiled_source():
    return importlib.util.find_spec("rixa._rixa_torch") is not None


def load_pmix_wheel():
    libdir = Path(__file__).resolve().parent.parent / "rixa.libs"

    candidates = list(libdir.glob("libpmix*.so*"))
    if not candidates:
        raise RuntimeError("PMIx not found")
    lib = sorted(candidates)[0]
    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def _get_ext():
    global _ext
    if _ext is not None:
        return _ext

    if importlib.util.find_spec("rixa._rixa_torch"):
        import rixa._rixa_torch as m

        _ext = m
        return _ext

    src_dir = os.path.join(os.path.dirname(__file__), "bindings")
    try:
        load_pmix_wheel()
    except RuntimeError:
        warnings.warn("Using global pmix to compile torch extension", stacklevel=2)

    _ext = cpp_ext.load(
        name="_rixa_torch",
        sources=[
            os.path.join(src_dir, "Rixa_pmix.cpp"),
            os.path.join(src_dir, "Rixa_C10.cpp"),
        ],
        extra_cflags=[
            "-O3",
        ],
        extra_include_paths=[
            src_dir,
        ],
        verbose=False,
    )
    return _ext


def get_pmix_store():
    return _get_ext().PMIxC10dStore


def init_process_group(backend="gloo", *args, gpu_assign_method="local_rank", **kwargs):
    store = get_pmix_store()()
    rank = store.rank()
    world = store.world_size()
    if backend == "nccl":
        match gpu_assign_method:
            case "local_rank":
                local_rank = store.local_rank()
                torch.cuda.set_device(local_rank)
            case "none":
                pass
            case _:
                raise ValueError(
                    "Wrong value for gpu_assign_method. Supported: local_rank, none"
                )

    dist.init_process_group(
        backend, *args, **kwargs, store=store, rank=rank, world_size=world
    )
