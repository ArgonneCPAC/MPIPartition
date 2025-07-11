#!/usr/bin/env python

"""Tests for `mpipartition.distribute` function"""

import numpy as np
import pytest

from mpipartition import S2Partition, s2_distribute


def create_and_distribute(N: int, equal_area: bool, iterations: int) -> None:
    partition = S2Partition(equal_area=equal_area)
    np.random.seed(partition.rank)

    phi = np.random.uniform(0, 2 * np.pi, N)
    theta = np.random.uniform(0, np.pi, N)
    id = np.arange(N, dtype=np.int64) + partition.rank * N

    data = {"phi": phi, "theta": theta, "id": id}
    data = s2_distribute(partition, data, all2all_iterations=iterations)

    valid = np.ones(len(data["phi"]), dtype=np.bool_)
    valid &= data["phi"] >= partition.phi_extent[0]
    valid &= data["phi"] < partition.phi_extent[1]
    valid &= data["theta"] >= partition.theta_extent[0]
    valid &= data["theta"] < partition.theta_extent[1]

    assert np.all(valid)


@pytest.mark.mpi
def test_s2_distribute_equal_area() -> None:
    create_and_distribute(100, equal_area=True, iterations=1)
    create_and_distribute(100, equal_area=True, iterations=2)
    create_and_distribute(100, equal_area=True, iterations=4)


@pytest.mark.mpi
def test_s2_distribute_unequal_area() -> None:
    create_and_distribute(100, equal_area=False, iterations=1)
    create_and_distribute(100, equal_area=False, iterations=2)
    create_and_distribute(100, equal_area=False, iterations=4)
