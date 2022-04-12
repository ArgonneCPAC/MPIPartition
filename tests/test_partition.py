#!/usr/bin/env python

"""Tests for `mpipartition` package."""

import pytest
from mpipartition import Partition
import numpy as np


@pytest.mark.xfail
def test_partition_0d():
    partition = Partition(0)


def _partition(dim):
    partition = Partition(dim)
    assert partition.dimensions == dim
    assert np.array(partition.decomposition).ndim == 1
    for i in range(dim):
        assert partition.origin[i] >= 0.0
        assert partition.origin[i] < 1.0
        assert partition.extent[i] > 0.0
        assert partition.extent[i] <= 1.0

    assert np.array(partition.ranklist).ndim == dim
    for i in range(dim):
        assert np.array(partition.ranklist).shape[i] == partition.decomposition[i]


@pytest.mark.mpi
def test_partition_1d():
    _partition(1)


@pytest.mark.mpi
def test_partition_2d():
    _partition(2)


@pytest.mark.mpi
def test_partition_3d():
    _partition(3)
