S2 Partitioning
===============

.. currentmodule:: mpipartition

The :class:`S2Partition` class will create a equal-area decomposition of a
spherical shell using the number of available MPI ranks. After initialization,
the instance contains information about the decomposition and the local rank
coordinates.

The decomposition follows the `"Recursive Zonal Equal Area Sphere Partitioning"
<https://eqsp.sourceforge.net/>`_ algorithm by Paul Leopardi. The spherical
shell will be divided into two polar caps (first and last MPI rank), and a
number of rings with a variable of segments in between. All "cuts" are at
constant phi or theta, which simplifies the implementation of ghost zones
between neighbors.

.. code-block:: python

    from mpipartition import S2Partition

    # partitioning S2 among the available ranks
    partition = Partition()

    # print theta and phi extent of all ranks:
    print(
        f"Rank {partition.rank}:\n"
        f"  theta: [{partition.theta_extent[0]:5.3f}, {partition.theta_extent[1]:5.3f}]\n"
        f"  phi  : [{partition.phi_extent[0]:5.3f}, {partition.phi_extent[1]:5.3f}]\n"
        f"  area : {partition.area:5.3f}"
    )

    # print size of this rank (as fraction of unit-cube).
    # Note: the extent of each rank will be the same
    if partition.rank == 0:
        print(partition.extent)

Example Partitions
------------------

You can use the ``mpipartition-s2`` executable to obtain the decomposition
information for a given number of ranks (and visualize the decomposition). Here
are some examples for 5, 10 and 100 ranks.

5 MPI ranks
^^^^^^^^^^^

.. figure:: figures/s2_partition_5ranks.svg
   :width: 600
   :align: center
   :alt: S2 partitioning with 5 ranks

   Segmentation of S2 with 5 MPI ranks.

.. code-block::

   Segmentation statistics for 5 ranks:
   polar cap angle: 0.927295
   number of rings: 1
      ring   0:   3 segments between theta=[0.927295, 2.214297]]
   Segment area imbalance:
      max/min: 1.000000
      max/avg: 1.000000
   Total edge/area ratio: 2.214498

10 MPI ranks
^^^^^^^^^^^^

.. figure:: figures/s2_partition_10ranks.svg
   :width: 600
   :align: center
   :alt: S2 partitioning with 10 ranks

   Segmentation of S2 with 10 MPI ranks.

.. code-block::

   Segmentation statistics for 10 ranks:
   polar cap angle: 0.643501
   number of rings: 2
      ring   0:   4 segments between theta=[0.643501, 1.570796]]
      ring   1:   4 segments between theta=[1.570796, 2.498092]]
   Segment area imbalance:
      max/min: 1.000000
      max/avg: 1.000000
   Total edge/area ratio: 3.380669

100 MPI ranks
^^^^^^^^^^^^^

.. figure:: figures/s2_partition_100ranks.svg
   :width: 600
   :align: center
   :alt: S2 partitioning with 100 ranks

   Segmentation of S2 with 100 MPI ranks.

.. code-block::

   Segmentation statistics for 100 ranks:
   polar cap angle: 0.200335
   number of rings: 8
      ring   0:   6 segments between theta=[0.200335, 0.535527]]
      ring   1:  11 segments between theta=[0.535527, 0.876298]]
      ring   2:  15 segments between theta=[0.876298, 1.223879]]
      ring   3:  17 segments between theta=[1.223879, 1.570796]]
      ring   4:  17 segments between theta=[1.570796, 1.917713]]
      ring   5:  15 segments between theta=[1.917713, 2.265295]]
      ring   6:  11 segments between theta=[2.265295, 2.606066]]
      ring   7:   6 segments between theta=[2.606066, 2.941258]]
   Segment area imbalance:
      max/min: 1.000000
      max/avg: 1.000000
   Total edge/area ratio: 11.206372

S2 Distribution Algorithms
==========================

Processing large datasets on multiple MPI ranks requires to distribute the data
among the processes. The ``mpipartition`` package contains the following
functions for data on the sphere:

.. autosummary::
   :nosignatures:

   s2_distribute
   s2_overload

Distribution/Overload Examples
------------------------------

In the following example, we generate 100 randomly positioned points per rank
and then distribute them according to the angular coordinates.

.. code-block:: python

   from mpipartition import S2Partition
   from mpipartition import s2_distribute, s2_overload

   # decompose a sphere with the available MPI ranks (equal area)
   partition = S2Partition()

   # number of random particles per rank
   n_local = 100

   # randomly distributed particles in a cube spanning [-1, 1]^3
   data = {
       "x": np.random.uniform(-1, 1, n_local),
       "y": np.random.uniform(-1, 1, n_local),
       "z": np.random.uniform(-1, 1, n_local),
       "id": n_local * partition.rank + np.arange(n_local),
       "rank": np.ones(n_local) * partition.rank
   }

   # calculate angular coordinates
   data['theta'] = np.arccos(data['z'])
   data['phi'] = np.arctan2(data['y'], data['x']) + np.pi

   # assign to rank by position
   data_distributed = s2_distribute(partition, data)


Now, we overload the partitions by 0.1 radians:

.. code-block:: python

   data_overloaded = s2_overload(partition, data_distributed, 0.1)


References
----------

S2Partition
^^^^^^^^^^^
.. autoclass:: S2Partition
   :members:

s2_distribute
^^^^^^^^^^^^^
.. autofunction:: s2_distribute

s2_overload
^^^^^^^^^^^
.. autofunction:: s2_overload

S2 decomposition visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. click:: mpipartition.scripts.s2_prediction:cli
    :prog: mpipartition-s2
    :nested: full
