# Rixa

Rixa (Runtime Initialization by pmiX Adoption) is a high-performance library that provides a unified and efficient way to bootstrap distributed PyTorch jobs.
It leverages PMIx 5.0 to seamlessly launch PyTorch workloads on large-scale HPC clusters, eliminating the need to manually specify the master IP address and port.


## Why Rixa?
Standard PyTorch bootstrapping (TCP/File-store) is built for cloud portability but often
struggles with scale and reliability on bare-metal HPC clusters.
`rixa` bypasses these overheads by using PMIx as a native high-performance key-value store, providing

1. Zero-config Launching: No master IP/Port orchestration required.

2. HPC Native: Leverages existing Slurm, Flux, or OpenMPI environments.

3. ABI Compatibility: Built on PMIx 5.0+ (ABI stable), ensuring portability across different MPI/PMIx versions.

## Instalation
In order to install one needs to specify the version of the library. Currently two versions are supported, `pytorch` and `nvshmem`.
They can be easily installed with
```bash
# Install for PyTorch support
pip install "rixa[pytorch]"

# Install for NVSHMEM support
pip install "rixa[nvshmem]"
```

## Usage (PyTorch)
One can use `rixa` to start pytorch distributed job with one simple line
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

## Usage (NVSHMEM)
Nvshmem usage is very simillar to the `pytorch` usage, one needs to use a thin wrapper around the native `nvshmem` init.
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

## Development
Library can be compiled from source with standard `setuptools` and requires PMIx library supporting version >5.0.
Development of the package is managed with `pixi` that can be used to also bring all necessary libraries for testing and development.
It can be locally tested using `prrte` and more recently with OpenMPI provided by `pixi`.
To build and test the library one can use commands provided in `pixi.toml`.
```
pixi run build  #downloads everything and builds
pixi run test   #launches pytest and uses openmpi>5.0 to launch the test
pixi run ci     #does both at the same time
```
Alternatively, one can use `pixi build` to compile a conda package.

## Roadmap

 - [x] Support for NVSHMEM (pytorch, cupy)
 - [ ] Support for JAX

## PMIx
One of the most common ways to bootstrap massive distributed programs is to use Process Management Interface (exascale), PMIx for short. It was designed
to launch massive MPI jobs accross enourmous HPC clusters, some achieving exascale performance.
It is integrated with the SLURM job scheduler and can selected to be a default launching
mechanism. Moreover, the 5th version it is ABI compatible. Hence, it avoids various problems related to the MPI-based programs (for example, `mpi4py`
needs to be compiled against specific MPI version). This makes this particular approach promising to achieve a platform-independent launching mechanism
for PyTorch distributed jobs on modern HPC clusters.

