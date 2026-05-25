rixa
==================


This documentation describes the ``rixa`` package, a simple wrapper around the ``pmix`` library used to start processes on various
supercomputers. It is natively usedy the ``mpi`` runtimes and is integrated into the ``slurm`` and ``flux`` schedulers. Due to the
native intagration on most HPC clusters it provides much easier setup compared to the master adress/port setup that is common
for distributed ``pytorch`` jobs started with the TCP bootstrap method.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   
   installation
   usage
   api
