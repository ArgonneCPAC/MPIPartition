#!/usr/bin/env python

"""Tests for `mpipartition` package."""


import pytest
from mpipartition import Partition, distribute, overload
import numpy as np


def _overloading(dimensions, n, ol):
    assert dimensions < 7
    labels = "xyzuvw"[:dimensions]

    partition = Partition(dimensions)
    for i in range(dimensions):
        assert ol < partition.extent[i]
    rank = partition.rank
    nranks = partition.nranks

    np.random.seed(rank)
    data = {
        x: np.random.uniform(0, 1, n) * partition.extent[i] + partition.origin[i]
        for i, x in enumerate(labels)
    }

    # exchange global data for verification
    global_data = {}
    for k in data.keys():
        global_data[k] = np.concatenate(partition.comm.allgather(data[k]))

    # mark origin of data
    data["s"] = rank * np.ones(n, dtype=np.uint16)
    data = overload(partition, 1.0, data, ol, labels)

    # did we give away any of our data?
    assert np.sum(data["s"] == rank) == n

    # unwrap periodic boundaries
    for i, x in enumerate(labels):
        overload_lo = partition.origin[i] - ol
        overload_hi = partition.origin[i] + partition.extent[i] + ol
        data[x][data[x] < overload_lo] += 1.0
        data[x][data[x] >= overload_hi] -= 1

    # check that we only have data within our overload
    for i, x in enumerate(labels):
        overload_lo = partition.origin[i] - ol
        overload_hi = partition.origin[i] + partition.extent[i] + ol
        assert np.all(data[x] >= overload_lo)
        assert np.all(data[x] < overload_hi)

    # cross-validate with global data
    mask_should_have = np.ones_like(global_data[labels[0]], dtype=np.bool_)
    for i, x in enumerate(labels):
        overload_lo = partition.origin[i] - ol
        overload_hi = partition.origin[i] + partition.extent[i] + ol
        # unwrap periodic boundary
        global_data[x][global_data[x] < overload_lo] += 1.0
        global_data[x][global_data[x] >= overload_hi] -= 1
        mask_should_have &= global_data[x] >= overload_lo
        mask_should_have &= global_data[x] < overload_hi

    assert np.sum(mask_should_have) == len(data[labels[0]])


@pytest.mark.mpi
def test_1d():
    _overloading(1, 1000, 0.1)
    _overloading(2, 100, 0.1)
    _overloading(3, 10, 0.05)
    _overloading(4, 2, 0.05)
