MPIPartition
============


.. image:: https://img.shields.io/pypi/v/mpipartition.svg
   :target: https://pypi.python.org/pypi/mpipartition

.. image:: https://github.com/ArgonneCPAC/MPIPartition/actions/workflows/pypi.yml/badge.svg
   :target: https://github.com/ArgonneCPAC/MPIPartition/actions/workflows/pypi.yml

.. image:: https://github.com/ArgonneCPAC/MPIPartition/actions/workflows/sphinx.yml/badge.svg
   :target: https://github.com/ArgonneCPAC/MPIPartition/actions/workflows/sphinx.yml

A python module for MPI volume decomposition and particle distribution


* Free software: MIT license
* Documentation: https://argonnecpac.github.io/MPIPartition
* Repository: https://github.com/ArgonneCPAC/MPIPartition


Features
--------

* Cartesian partitioning of a cubic volume among available MPI ranks
* distributing particle-data among ranks to the corresponding subvolume
* overloading particle-data at rank boundaries
* exchaning particle-data according to a "owner"-list of keys per rank


Installation
------------

Installing from the PyPI repository:

.. code-block:: bash

   pip install mpipartition

Installing the development version from the GIT repository

.. code-block:: bash

   git clone https://github.com/ArgonneCPAC/mpipartition.git
   cd mpipartition
   python setup.py develop


Requirements
------------

* Python >= 3.7
* `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_: MPI for Python
* `numpy <https://numpy.org/>`_: Python array library


Basic Usage
-----------
Check the `documentation <https://argonnecpac.github.io/MPIPartition>`_ for
an in-depth explanation / documentation.

.. code-block:: python

   # this code goes into mpipartition_example.py

   from mpipartition import Partition, distribute, overload
   import numpy as np

   # create a partition of the unit cube with available MPI ranks
   box_size = 1.
   partition = Partition()

   if partition.rank == 0:
       print(f"Number of ranks: {partition.nranks}")
       print(f"Volume decomposition: {partition.decomposition}")

   # create random data
   nparticles_local = 1000
   data = {
       "x": np.random.uniform(0, 1, nparticles_local),
       "y": np.random.uniform(0, 1, nparticles_local),
       "z": np.random.uniform(0, 1, nparticles_local)
   }

   # distribute data to ranks assigned to corresponding subvolume
   data = distribute(partition, box_size, data, ('x', 'y', 'z'))

   # overload "edge" of each subvolume by 0.05
   data = overload(partition, box_size, data, 0.05, ('x', 'y', 'z'))

This code can then be executed with ``mpi``:

.. code-block:: bash

   mpirun -n 10 python mpipartition_example.py

--------

A more applied example, using halo catalogs from a
`HACC <https://cpac.hep.anl.gov/projects/hacc/>`_ cosmological simulation (in
the `GenericIO <https://git.cels.anl.gov/hacc/genericio>`_ data format):

.. code-block:: python

   from mpipartition import Partition, distribute, overload
   import numpy as np
   import pygio

   # create a partition with available MPI ranks
   box_size = 64.  # box size in Mpc/h
   partition = Partition(3)  # by default, the dimension is 3

   # read GenericIO data in parallel
   data = pygio.read_genericio("m000p-499.haloproperties")

   # distribute
   data = distribute(partition, box_size, data, [f"fof_halo_center_{x}" for x in "xyz"])

   # mark "owned" data with rank (allows differentiating owned and overloaded data)
   data["status"] = partition.rank * np.ones(len(data["fof_halo_center_x"]), dtype=np.uint16)

   # overload by 4Mpc/h
   data = overload(partition, box_size, data, 4., [f"fof_halo_center_{x}" for x in "xyz"])

   # now we can do analysis such as 2pt correlation functions (up to 4Mpc/h)
   # or neighbor finding, etc.
