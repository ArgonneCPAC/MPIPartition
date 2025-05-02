#!/usr/bin/env python

"""Tests for `mpipartition` package."""

import numpy as np
import pytest

from mpipartition import Partition, overload
from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init()
nranks = MPI.COMM_WORLD.Get_size()


def _overloading(dimensions: int, n: int, ol: float) -> None:
    assert dimensions < 7
    labels = "xyzuvw"[:dimensions]
    coord_keys = [x for x in labels]

    partition = Partition(dimensions)
    rank = partition.rank
    nranks = partition.nranks
    np.random.seed(rank)

    # generate data within our partition
    data: dict[str, np.ndarray] = {
        x: np.random.uniform(0, 1, n) * partition.extent[i] + partition.origin[i]
        for i, x in enumerate(labels)
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
    data = overload(partition, 1.0, data, ol, coord_keys=[x for x in labels])

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

    for i in range(nranks):
        if i == rank and not np.sum(mask_should_have) == len(data[labels[0]]):
            print(
                "ERROR ON RANK",
                rank,
                partition.origin,
                partition.extent,
                ol,
            )
            ids_should_have = global_data["id"][mask_should_have]
            ids_have = data["id"]
            missing_idx = ~np.isin(ids_should_have, ids_have)
            print("MISSING IDS:", ids_should_have[missing_idx])
            print("MISSING POS:", global_data["x"][mask_should_have][missing_idx])
            print("MISSING_RANK:", global_data["s"][mask_should_have][missing_idx])
            print("", flush=True)
        partition.comm.Barrier()
    assert np.sum(mask_should_have) == len(data[labels[0]])


def _check_0overload(dimensions: int, n: int) -> None:
    assert dimensions < 7
    labels = "xyzuvw"[:dimensions]
    coord_keys = [x for x in labels]

    partition = Partition(dimensions)

    rank = partition.rank

    np.random.seed(rank)
    data: dict[str, np.ndarray] = {
        x: np.random.uniform(0, 1, n) * partition.extent[i] + partition.origin[i]
        for i, x in enumerate(labels)
    }
    data["s"] = rank * np.ones(n, dtype=np.uint16)

    data_ol = overload(partition, 1.0, data, 0.0, coord_keys)
    # Check that we haven't changed any of the data
    assert len(data_ol[labels[0]] == n)
    assert np.all(data["s"] == rank)


def _overloading_struct(dimensions: int, n: int, ol: float) -> None:
    assert dimensions < 7
    labels = "xyzuvw"[:dimensions]
    coord_keys = [x for x in labels]

    n_struct = int(n / 4)

    partition = Partition(dimensions)
    rank = partition.rank
    nranks = partition.nranks
    np.random.seed(rank)

    # generate data within our partition
    data: dict[str, np.ndarray] = {
        x: np.random.uniform(0, 1, n) * partition.extent[i] + partition.origin[i]
        for i, x in enumerate(labels)
    }
    # unique id
    data["id"] = np.arange(n, dtype=np.uint64) + rank * n
    # mark origin of data
    data["s"] = rank * np.ones(n, dtype=np.uint16)
    # structure tag
    data["struct"] = np.random.randint(0, n_struct, n) + rank * n_struct

    # exchange global data for verification
    global_data = {}
    for k in data.keys():
        global_data[k] = np.concatenate(partition.comm.allgather(data[k]))
        assert len(global_data[k]) == nranks * n

    # overload data
    data = overload(partition, 1.0, data, ol, coord_keys, structure_key="struct")

    # did we give away any of our data?
    assert np.sum(data["s"] == rank) == n

    # check that if we have any obj of a "st", that we have every obj of that "st"
    present_structs = np.unique(data["struct"])
    needed_objs = global_data["id"][np.isin(global_data["struct"], present_structs)]
    missing_objs = np.all(np.isin(needed_objs, data["id"]))
    assert np.all(np.isin(needed_objs, data["id"]))


@pytest.mark.mpi
def test_1d_large_ol() -> None:
    partition = Partition(1)
    if np.min(partition.decomposition) == 1:
        pytest.xfail("invalid number of MPI ranks for overload")
    ol = 0.9 * 1 / np.max(partition.decomposition)
    _check_0overload(1, 1000)
    _overloading(1, 1000, ol)


@pytest.mark.mpi
def test_1d() -> None:
    partition = Partition(1)
    if np.min(partition.decomposition) == 1:
        pytest.xfail("invalid number of MPI ranks for overload")
    ol = 0.49 * 1 / np.max(partition.decomposition)
    _check_0overload(1, 1000)
    _overloading(1, 1000, ol)
    _overloading_struct(1, 1000, ol)


@pytest.mark.mpi
def test_2d() -> None:
    partition = Partition(2)
    if np.min(partition.decomposition) == 1:
        pytest.xfail("invalid number of MPI ranks for overload")
    ol = 0.49 * 1 / np.max(partition.decomposition)
    _check_0overload(2, 100)
    _overloading(2, 100, ol)
    _overloading_struct(2, 100, ol)


@pytest.mark.mpi
def test_3d() -> None:
    partition = Partition(3)
    if np.min(partition.decomposition) == 1:
        pytest.xfail("invalid number of MPI ranks for overload")
    ol = 0.49 * 1 / np.max(partition.decomposition)
    _check_0overload(3, 10)
    _overloading(3, 10, ol)
    _overloading_struct(3, 10, ol)


@pytest.mark.mpi
def test_4d() -> None:
    partition = Partition(4)
    if np.min(partition.decomposition) == 1:
        pytest.xfail("invalid number of MPI ranks for overload")
    ol = 0.49 * 1 / np.max(partition.decomposition)
    _check_0overload(4, 4)
    _overloading(4, 4, ol)
    _overloading_struct(4, 4, ol)
