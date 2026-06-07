# Rixa

> [!WARNING]
> This library is in an experimental phase and may be buggy.
> Expect breaking changes between versions.

Rixa (Runtime Initialization by pmiX Adoption) is a high-performance library that provides a unified and efficient way to bootstrap distributed PyTorch jobs.
It leverages PMIx 5.0 to seamlessly launch PyTorch workloads on large-scale HPC clusters, eliminating the need to manually specify the master IP address and port.


## Why Rixa?
Standard PyTorch bootstrapping (TCP/File-store) is built for cloud portability but often
struggles with scale and reliability on bare-metal HPC clusters.
`rixa` bypasses these overheads by using PMIx as a native high-performance key-value store, providing

1. Zero-config Launching: No master IP/Port orchestration required.

2. HPC Native: Leverages existing Slurm, Flux, or OpenMPI environments.

3. ABI Compatibility: Built on PMIx 5.0+ (ABI stable), ensuring portability across different MPI/PMIx versions.

## Installation
> [!NOTE]
> For detailed installation instructions, see the [wiki](https://github.com/wesenheit/rixa/wiki/Installation).


In order to install one needs to specify the version of the library. Currently two versions are supported, `pytorch` and `nvshmem`.
They can be easily installed with
```bash
# Install for PyTorch support
pip install "rixa[pytorch]"

# Install for NVSHMEM support
pip install "rixa[nvshmem]"
```
## Usage
> [!NOTE]
> For more use cases, see the [wiki](https://github.com/wesenheit/rixa/wiki/Quick-Start).


### PyTorch
One can use `rixa` to start PyTorch distributed job with one simple line
```python
import rixa
rixa.pytorch.init_process_group(pytorch_argument1, pytorch_argument2, keyword2=pytorch_parameter)
```
Jobs can be launched with any PMIx 5.0-compatible plugin, starting with `prrte`, some MPI implementations (OpenMPI 5.0),
native job launcher plugins to SLURM or Flux. Example:
```bash
prterun -n 16 python3 -c "import rixa; rixa.pytorch.init_process_group(); import torch; print(torch.distributed.get_rank())"
```
Remember to manually finalize the backend!
```python
torch.distributed.destroy_process_group()
```

### NVSHMEM
Nvshmem usage is very similar to the `pytorch` usage, one needs to use a thin wrapper around the native `nvshmem` init.
Example:
```python
import rixa
from cuda.core import Device

store = rixa.PMIxStore(30) #manualy specify the PMIx backend with manual timeout, can become handy to set the device
dev = Device(0) #first device or just set based on the use case
rixa.nvshmem.init(dev, store) #device, store
```
Remember to manually finalize the backend!
```python
nvshmem.finalize()
```

## Roadmap

 - [x] Support for NVSHMEM (pytorch, cupy)
 - [ ] Support for JAX

