MPI Partitioning
================

.. currentmodule:: mpipartition

The :class:`Partition` class will create a volume decomposition using the
number of available MPI ranks. After initialization, the instance contains
information about the decomposition and the local rank coordinates.

.. code-block:: python

   from mpipartition import Partition

   # partitioning a box among the available ranks
   partition = Partition()

   # print how the volume has been partitioned (from rank 0):
   if partition.rank == 0:
       print(partition.decomposition)

   # print coordinate of all ranks:
   print(partition.rank, partition.coordinates)

   # print size of this rank (as fraction of unit-cube).
   # Note: the extent of each rank will be the same
   if partition.rank == 0:
       print(partition.extent)





MPI Distribution Algorithms
===========================

Processing large datasets on multiple MPI ranks requires to distribute the data
among the processes. The ``mpipartition`` package contains the following
functions that abstract this task in different use-cases:

.. autosummary::
   :nosignatures:

   distribute
   overload
   exchange

Examples
--------

In the following example, we generate 100 randomly positioned points per rank
and then distribute them according to the positions

.. code-block:: python

   from mpipartition import Partition
   from mpipartition import distribute, overload, exchange

   # assuming a cube size of 1.
   box_size = 1.0

   # partitioning a box with the available MPI ranks
   partition = Partition()

   # number of random particles per rank
   n_local = 100

   # randomly distributed particles in the unit cube
   data = {
       "x": np.random.uniform(0, 1, n_local),
       "y": np.random.uniform(0, 1, n_local),
       "z": np.random.uniform(0, 1, n_local),
       "id": n_local * partition.rank + np.arange(n_local),
       "rank": np.ones(n_local) * partition.rank
   }

   # assign to rank by position
   data_distributed = distribute(partition, box_size, data, ('x', 'y', 'z'))

   # make sure we still have all particles
   n_local_distributed = len(data_distributed['x'])
   n_global_distributed = partition.comm.reduce(n_local_distributed)
   if partition.rank == 0:
       assert n_global_distributed == n_local * partition.nranks

   # validate that each particle is in local extent
   bbox = np.array([
      np.array(partition.origin),
      np.array(partition.origin) + np.array(partition.extent)
   ]).T
   is_valid = np.ones(n_local_after, dtype=np.bool_)
   for i, x in enumerate(['xyz']):
       is_valid &= data_distributed[x] >= bbox[i, 0]
       is_valid &= data_distributed[x] < bbox[i, 1]
   assert np.all(is_valid)

Now, we overload the partitions by 0.1

.. code-block:: python

   data_overloaded = overload(partition, box_size, data_distributed, 0.1, ('x', 'y', 'z'))


Sometimes, the destination of a particle is given by a key, not by the position
(e.g. for a merger-tree, we want the progenitors to be on the same rank as the
descendant, even if they cross the rank boundaries). We can then use the
``exchange`` function as follows:

.. code-block:: python

   # create a list of particle ids that we want to have on the local rank
   my_keys = n_local * partition.rank + np.arange(n_local)

   # since in our example, particles can be further away than 1 neighboring
   # rank, we directly do an all2all exchange:
   data_exchanged = exchange(partition, data_distributed, 'id', my_keys, do_all2all=True)

   # we should have the same particles as we started with! Let's check
   s = np.argsort(data_exchanged['id'])
   for k in data_exchanged.keys():
       data_exchanged[k] = data_exchanged[k][s]

   n_local_exchanged = len(data_exchanged['x'])
   assert n_local_exchanged == n_local
   for k in data.keys():
       assert np.all(data[k] == data_exchanged[k])


References
----------

Partition
^^^^^^^^^
.. autoclass:: Partition
   :members:

distribute
^^^^^^^^^^
.. autofunction:: distribute

overload
^^^^^^^^
.. autofunction:: overload

exchange
^^^^^^^^
.. autofunction:: exchange