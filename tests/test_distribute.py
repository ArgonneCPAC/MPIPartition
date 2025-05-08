#!/usr/bin/env python

"""Tests for `mpipartition.distribute` function"""

import numpy as np
import pytest

from mpipartition import Partition, distribute


def create_and_distribute(dimensions: int, N: int, iterations: int) -> None:
    partition = Partition(dimensions)
    assert dimensions < 7
    assert dimensions > 0
    labels = "xyzuvw"[:dimensions]
    # as a list of characters
    coord_keys = [x for x in labels]

    data = {x: np.random.uniform(0, 1, N) for x in labels}
    data = distribute(partition, 1.0, data, coord_keys, all2all_iterations=iterations)

    valid = np.ones(len(data[labels[0]]), dtype=np.bool_)
    for i, label in enumerate(labels):
        valid &= data[label] >= partition.origin[i]
        valid &= data[label] < partition.origin[i] + partition.extent[i]
    local_all_valid = np.all(valid)
    global_all_valid = np.all(partition.comm.allgather(local_all_valid))

    assert global_all_valid
    # if partition.rank == 0:
    #     print(f"Passed for dim={dimensions} and {N} particles per rank: {global_all_valid}")


@pytest.mark.mpi
def test_1d() -> None:
    create_and_distribute(1, 1000, 1)
    create_and_distribute(1, 1000, 2)
    create_and_distribute(1, 1000, 4)


@pytest.mark.mpi
def test_2d() -> None:
    create_and_distribute(2, 100, 1)
    create_and_distribute(2, 100, 2)
    create_and_distribute(2, 100, 4)


@pytest.mark.mpi
def test_3d() -> None:
    create_and_distribute(3, 10, 1)
    create_and_distribute(3, 10, 2)
    create_and_distribute(3, 10, 4)


@pytest.mark.mpi
def test_4d() -> None:
    create_and_distribute(4, 5, 1)
    create_and_distribute(4, 5, 2)
    create_and_distribute(4, 5, 4)
