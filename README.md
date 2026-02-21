# Foehn

Foehn is a high-performance library that provides a unified and efficient way to bootstrap distributed PyTorch jobs.
It leverages PMIx 5.0 to seamlessly launch PyTorch workloads on large-scale HPC clusters, eliminating the need to manually specify the master IP address and port.

## Alternatives
In general, PyTorch uses TCP or file-based solutions to broadcast the initial information used in the bootstrap process. Both of those
methods are designed to work on almost all machines (starting from bare-metal clusters and ending with cloud enviroments).
Nevertherless, neither are they designed for high performance, nor are they designed to
leverage software stack on modern HPC clusters. Especially, they are not designed to use usuall HPC stack.

## PMIx
One of the most common ways to bootstrap massive distributed programs is to use Process Management Interface (exascale), PMIx for short. It was designed
to launch massive MPI jobs accross enourmous HPC clusters, some achieving exascale performance.
It is integrated with the SLURM job scheduler and can selected to be a default launching
mechanism. Moreover, since the 5th version it is ABI compatible. Hence, it avoids various problems related to the MPI-based programs (for example, mpi4py
needs to be compiled against specific MPI version). This makes this particular approach promising to achieve a platform-independent launching mechanism
for PyTorch distributed jobs on modern HPC clusters.

## Usage

Just run
```python
import foehn
foehn.pytorch.init_process_group(pytorch_argument1, pytorch_argument2, keyword2=pytorch_parameter)
```

## Installation
Library can be compiled from soure using meson with dependences manged with uv. It requires PMIx library supporting version >5.0.
It can be locally tested using [prrte](https://github.com/openpmix/prrte). To build and test it one can use commands provided in justfile (requires just installed).


```
uv sync # download dependencies
just setup # setup build library
just build # compile code with meson
just test  # test the library, requires prrte
```


## Roadmap

 - [ ] Support for NVSCHMEM (pytorch)
 - [ ] Support for JAX
