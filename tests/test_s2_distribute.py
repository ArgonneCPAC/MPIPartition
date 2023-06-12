#!/usr/bin/env python

"""Tests for `mpipartition.distribute` function"""

import numpy as np
import pytest

from mpipartition import S2Partition, s2_distribute


def create_and_distribute(N, equal_area):
    partition = S2Partition(equal_area=equal_area)
    np.random.seed(partition.rank)

    phi = np.random.uniform(0, 2 * np.pi, N)
    theta = np.random.uniform(0, np.pi, N)
    id = np.arange(N, dtype=np.int64) + partition.rank * N

    data = {"phi": phi, "theta": theta, "id": id}
    data = s2_distribute(partition, data)

    valid = np.ones(len(data["phi"]), dtype=np.bool_)
    valid &= data["phi"] >= partition.phi_extent[0]
    valid &= data["phi"] < partition.phi_extent[1]
    valid &= data["theta"] >= partition.theta_extent[0]
    valid &= data["theta"] < partition.theta_extent[1]

    assert np.all(valid)


@pytest.mark.mpi
def test_s2_distribute_equal_area():
    create_and_distribute(100, equal_area=True)


@pytest.mark.mpi
def test_s2_distribute_unequal_area():
    create_and_distribute(100, equal_area=False)
