from mpipartition import Partition

# partitioning a box among the available ranks (3d)
partition = Partition()
if partition.rank == 0:
    print(partition.dimensions)
    print(partition.decomposition)
    print(partition.extent)
    print(partition.neighbors)
    print(partition.ranklist)

# partitioning a box among the available ranks (2d)
partition = Partition(dimensions=2)
if partition.rank == 0:
    print(partition.dimensions)
    print(partition.decomposition)
    print(partition.extent)
    print(partition.neighbors)
    print(partition.ranklist)

# partitioning a box among the available ranks (1d)
partition = Partition(dimensions=1)
if partition.rank == 0:
    print(partition.dimensions)
    print(partition.decomposition)
    print(partition.extent)
    print(partition.neighbors)
    print(partition.ranklist)

# partitioning a box among the available ranks (4d)
partition = Partition(dimensions=4)
if partition.rank == 0:
    print(partition.dimensions)
    print(partition.decomposition)
    print(partition.extent)
    print(partition.neighbors)
    print(partition.ranklist)

