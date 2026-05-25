# Usage

There are two usecases supported in the current version of the library: `pytorch`  and `nvshmem`.
Versions can be installed with `rixa[torch]` and `rixa[nvshmem]` specifiers.

## Pytorch usage
In order to use `rixa` with `pytorch`-based programs one needs to simply initalize the group with

```python
import rixa
import torch
rixa.pytorch.init_process_group(pytorch_argument1, pytorch_argument2, keyword2=pytorch_parameter)

# DO SOME WORK
torch.distributed.destroy_process_group()
```

## NVSHMEM usage
In order to initialize `nvshmem` there is a need to specify more parameters.
First of all, there is a need to manually define the PMIx key-value store

```python
import rixa
from cuda.core import Device

store = rixa.PMIxStore(30) #manualy specify the PMIx backend with manual timeout
dev = Device(0) #first device or just set based on the use case
rixa.nvshmem.init(dev, store) #device, store

# DO SOME WORK

nvshmem.finalize()
```
