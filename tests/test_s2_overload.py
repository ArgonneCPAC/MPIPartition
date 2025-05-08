#!/usr/bin/env python

"""Tests for `mpipartition` package."""

import numpy as np
import pytest

from mpipartition import S2Partition, s2_overload
from mpi4py import MPI

nranks = MPI.COMM_WORLD.Get_size()


def _overloading(n: int, overload_angle: float, equal_area: bool) -> None:
    partition = S2Partition(equal_area=equal_area)

    rank = partition.rank
    nranks = partition.nranks

    my_phi_range = partition.phi_extent
    my_theta_range = partition.theta_extent
    np.random.seed(rank)

    # generate data within our partition
    data: dict[str, np.ndarray] = {
        "phi": np.random.uniform(
            my_phi_range[0],
            my_phi_range[1],
            n,
        ),
        "theta": np.random.uniform(
            my_theta_range[0],
            my_theta_range[1],
            n,
        ),
    }
    # unique id
    data["id"] = np.arange(n, dtype=np.uint64) + rank * n
    # mark origin of data
    data["s"] = rank * np.ones(n, dtype=np.uint16)

    # exchange global data for verification
    global_data = {}
    for k in data.keys():
        global_data[k] = np.concatenate(partition.comm.allgather(data[k]))
        assert len(global_data[k]) == nranks * n

    # overload data
    data = s2_overload(partition, data, overload_angle)

    # did we give away any of our data?
    assert np.sum(data["s"] == rank) == n

    # unwrap periodic boundaries along phi
    data["phi"][data["phi"] < my_phi_range[0] - overload_angle] += 2 * np.pi
    data["phi"][data["phi"] >= my_phi_range[1] + overload_angle] -= 2 * np.pi

    # check that we only have data within our overload
    assert np.all(data["phi"] >= my_phi_range[0] - overload_angle)
    assert np.all(data["phi"] < my_phi_range[1] + overload_angle)
    assert np.all(data["theta"] >= my_theta_range[0] - overload_angle)
    assert np.all(data["theta"] < my_theta_range[1] + overload_angle)

    # cross-validate with global data
    mask_should_have = np.ones_like(global_data["theta"], dtype=np.bool_)
    # unwrap periodic boundaries along phi
    global_data["phi"][global_data["phi"] < my_phi_range[0] - overload_angle] += (
        2 * np.pi
    )
    global_data["phi"][global_data["phi"] >= my_phi_range[1] + overload_angle] -= (
        2 * np.pi
    )
    mask_should_have &= global_data["phi"] >= my_phi_range[0] - overload_angle
    mask_should_have &= global_data["phi"] < my_phi_range[1] + overload_angle
    mask_should_have &= global_data["theta"] >= my_theta_range[0] - overload_angle
    mask_should_have &= global_data["theta"] < my_theta_range[1] + overload_angle

    for i in range(nranks):
        if i == rank and not np.sum(mask_should_have) == len(data["theta"]):
            print(
                "ERROR ON RANK",
                rank,
                my_phi_range,
                my_theta_range,
                overload_angle,
            )
            ids_should_have = global_data["id"][mask_should_have]
            ids_have = data["id"]
            missing_idx = ~np.isin(ids_should_have, ids_have)
            print("MISSING IDS:", ids_should_have[missing_idx])
            print("MISSING PHI:", global_data["phi"][mask_should_have][missing_idx])
            print("MISSING THETA:", global_data["theta"][mask_should_have][missing_idx])
            print("MISSING_RANK:", global_data["s"][mask_should_have][missing_idx])
            print("", flush=True)
        partition.comm.Barrier()
    assert np.sum(mask_should_have) == len(data["theta"])


@pytest.mark.mpi
def test_ovlerload_equal_area() -> None:
    _overloading(100, 0.2, equal_area=True)


@pytest.mark.mpi
def test_ovlerload_unequal_area() -> None:
    _overloading(100, 0.2, equal_area=False)
