#!/usr/bin/env python

"""Tests for `mpipartition` package."""


import pytest
from mpipartition import Partition, exchange
import numpy as np


def create_and_distribute(partition: Partition, N: int):
    data = {
        "id": np.arange(N, dtype=np.int32) + partition.rank * N,
        "origin": partition.rank * np.ones(N, dtype=np.int32),
    }
    # redistribute the data so we own all particles with id % nrank == our_rank
    my_keys = np.arange(
        partition.rank, partition.nranks * N, partition.nranks, dtype=np.int32
    )
    data_new = exchange(partition, data, key="id", local_keys=my_keys)

    assert np.all(np.isin(data_new["id"], my_keys))
    assert np.all(data_new["id"] % partition.nranks == partition.rank)
    assert len(data_new["id"]) == N


@pytest.mark.mpi
def test_exchange_10():
    partition = Partition(1, create_neighbor_topo=True)
    create_and_distribute(partition, 10)


@pytest.mark.mpi
def test_exchange_80():
    partition = Partition(1, create_neighbor_topo=True)
    create_and_distribute(partition, 80)


@pytest.mark.mpi
def test_exchange_101():
    partition = Partition(1, create_neighbor_topo=True)
    create_and_distribute(partition, 101)
