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


def _check_0overload(dimensions, n):
    assert dimensions < 7
    labels = "xyzuvw"[:dimensions]

    partition = Partition(dimensions)

    rank = partition.rank
    nranks = partition.nranks

    np.random.seed(rank)
    data = {
        x: np.random.uniform(0, 1, n) * partition.extent[i] + partition.origin[i]
        for i, x in enumerate(labels)
    }
    data["s"] = rank * np.ones(n, dtype=np.uint16)

    data_ol = overload(partition, 1.0, data, 0.0, labels)
    # Check that we haven't changed any of the data
    assert len(data_ol[labels[0]] == n)
    assert np.all(data["s"] == rank)


@pytest.mark.mpi
def test_1d():
    _check_0overload(1, 1000)
    _overloading(1, 1000, 0.1)


@pytest.mark.mpi
def test_2d():
    _check_0overload(2, 100)
    _overloading(2, 100, 0.1)


@pytest.mark.mpi
def test_3d():
    _check_0overload(3, 10)
    _overloading(3, 10, 0.05)


@pytest.mark.mpi
def test_4d():
    _check_0overload(4, 4)
    _overloading(4, 4, 0.05)
