# Installation

There are two main supported modes of installation: source distribution and prepared wheel. First mode requires
to have the `pmix` library in the development version newer than v5.0. Second model pulls the precompiled
wheel that is already shipping with the PMIx library.


## Source distribution

`rixa` can be compiled from source by running

```shell
pip install rixa --no-binary rixa

```
Alternativelly, one can configure the `uv` package manager not to use the binary

```toml
[tool.uv]
no-binary = ["rixa"]
```

There are few reasons why the source distribution way is prefered and encouraged:

1. Compaility with existing HPC stack - `rixa` requires PMIx-based launcher to initialize the processes. This
usually means that the PMIx stack is already installed. Linking against the existing version decreases various problems
with differnt PMIx versions.
2. Compiled wheel is configured with specific configuration to make the wheels leightweight and hence may differ from the
version that is used by the launcher.


Hence, performing source installation is enouraged, especially on multi-node clusters that expose PMIx versions directly.

## Wheel distribution
Some users may prefer to use the binary distribution to pull the PMIx library. There are only few reasons that the user
 may want to use the binary over the source distribution:

1. HPC cluster does not expose PMIx,
2. PMIx installation lacks development headers.

In both cases it is important to emphasize that the compiled version is PMIx v5.0.9. Hence, launchers compiled with ABI compatible
versions of the PMIx should run without problem.
