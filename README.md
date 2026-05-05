# Foehn

Foehn is a high-performance library that provides a unified and efficient way to bootstrap distributed PyTorch jobs.
It leverages PMIx 5.0 to seamlessly launch PyTorch workloads on large-scale HPC clusters, eliminating the need to manually specify the master IP address and port.

## Alternatives
In general, PyTorch uses TCP or file-based solutions to broadcast the initial information used in the bootstrap process. Both of those
methods are designed to work on almost all machines (starting from bare-metal clusters and ending with cloud enviroments).
Nevertherless, neither are they designed for high performance, nor are they designed to
leverage software stack on modern HPC clusters.

## PMIx
One of the most common ways to bootstrap massive distributed programs is to use Process Management Interface (exascale), PMIx for short. It was designed
to launch massive MPI jobs accross enourmous HPC clusters, some achieving exascale performance.
It is integrated with the SLURM job scheduler and can selected to be a default launching
mechanism. Moreover, since the 5th version it is ABI compatible. Hence, it avoids various problems related to the MPI-based programs (for example, `mpi4py`
needs to be compiled against specific MPI version). This makes this particular approach promising to achieve a platform-independent launching mechanism
for PyTorch distributed jobs on modern HPC clusters.

## Usage
One can use `foehn` to start pytorch distributed job with one simple line
```python
import foehn
foehn.pytorch.init_process_group(pytorch_argument1, pytorch_argument2, keyword2=pytorch_parameter)
```
Jobs can be launched with any PMIx 5.0-compatible plugin, starting with `prrte`, some MPI implementations (OpenMPI 5.0),
native job launcher plugins to SLURM or Flux. Example:
```
 prterun -n 16 python3 -c "import foehn; foehn.pytorch.init_process_group(); import torch; print(torch.distributed.get_rank())"
```



## Installation
Library can be compiled from source with standard `setuptools` and requires PMIx library supporting version >5.0.
Development of the package is manged with `pixi` that can be used to also bring all necessary libraries for testing and development.
It can be locally tested using `prrte` and more recently with OpenMPI provided by `pixi`.
To build and test the library one can use commands provided in `pixi.toml`.
```
pixi run build  #downloads everything and builds
pixi run test   #launches pytest and uses openmpi>5.0 to launch the test
pixi run ci     #does both at the same time
```
Alternativelly, one can use `pixi build` to compile a conda package.

## Roadmap

 - [ ] Support for NVSCHMEM (pytorch)
 - [ ] Support for JAX
