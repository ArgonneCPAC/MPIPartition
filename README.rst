MPIPartition
============


.. image:: https://img.shields.io/pypi/v/mpipartition.svg
        :target: https://pypi.python.org/pypi/mpipartition



A python module for MPI volume decomposition and particle distribution


* Free software: MIT license
* Documentation: https://argonnecpac.github.io/MPIPartition


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

* `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_: MPI for Python
* `numpy <https://numpy.org/>`_: Python array library