#!/usr/bin/env python

"""Tests for `mpipartition.distribute` function"""

import numpy as np
import pytest

from mpipartition import Partition, distribute


def create_and_distribute(dimensions, N):
    partition = Partition(dimensions)
    assert dimensions < 7
    assert dimensions > 0
    labels = "xyzuvw"[:dimensions]

    data = {x: np.random.uniform(0, 1, N) for x in labels}
    data = distribute(partition, 1.0, data, labels)

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
def test_1d():
    create_and_distribute(1, 1000)


@pytest.mark.mpi
def test_2d():
    create_and_distribute(2, 100)


@pytest.mark.mpi
def test_3d():
    create_and_distribute(3, 10)


@pytest.mark.mpi
def test_4d():
    create_and_distribute(4, 5)
