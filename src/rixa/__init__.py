import importlib.util
from .PMIx_core import PMIxStore
import os

is_strict = False  # We don't want to be strict maybe?
# Ninja has build lock, first process will acquire the lock


def is_distributed_env():
    return "PMIX_RANK" in os.environ


__all__ = [
    "PMIxStore",
]
if importlib.util.find_spec("torch") is not None:
    from . import pytorch

    if not (pytorch.is_compiled_source() or pytorch.is_compiled_lazy()):
        if is_distributed_env() and is_strict:
            raise RuntimeError(
                "Pytorch support and distributed enviroment is set up but noting is yet compiled!\n"
                'In order to compile the extension do python -c "import rixa"  '
            )
        else:
            pytorch.get_pmix_store()
    __all__ += ["pytorch"]


if importlib.util.find_spec("nvshmem") is not None:
    from . import nvshmem

    __all__ += ["nvshmem"]
