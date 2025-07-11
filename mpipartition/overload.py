import itertools
from typing import List, Union

import numpy as np
import numpy.typing as npt

from .partition import Partition

ParticleDataT = dict[str, np.ndarray]


def overload(
    partition: Partition,
    box_size: float,
    data: ParticleDataT,
    overload_length: float,
    coord_keys: List[str],
    *,
    structure_key: str | None = None,
    verbose: Union[bool, int] = False,
) -> ParticleDataT:
    """Copy data within an overload length to the neighboring ranks

    This method assumes that the volume cube is periodic and will wrap the data
    around the boundary interfaces.

    Parameters
    ----------
    partition:
        The MPI partition defining which rank should own which subvolume of the
        data

    box_size:
        the size of the full volume cube

    data:
        The treenode / coretree data that should be distributed

    overload_length:
        The thickness of the boundary layer that will be copied to the
        neighboring rank. Must be smaller than half the extent of the local
        subvolume (along any axis)

    coord_keys:
        The columns in `data` that define the position of the object

    structure_key:
        The column in `data` containing a structure ("group") tag. If provided,
        the data will be overloaded to include entire structures; ie when one
        object in a structure is overloaded, all other objects in that structure
        are sent as well. The column `data[structure_key]` should be of integer
        type, and any objects not belonging to a structure are assumed to have
        tag -1.

    verbose:
        If True, print summary statistics of the distribute. If > 1, print
        statistics of each rank (i.e. how much data each rank sends to every
        other rank).

    Returns
    -------
    data: TreeDataT
        The combined data of objects within the rank's subvolume as well as the
        objects within the overload region of neighboring ranks

    Notes
    -----

    The function does not change the objects' coordinates or alter any data.
    Objects that have been overloaded accross the periodic boundaries will still
    have the original positions. In case "local" coordinates are required, this
    will need to be done manually after calling this function.

    """
    assert len(coord_keys) == partition.dimensions
    for i in range(partition.dimensions):
        assert partition.decomposition[i] > 1  # currently can't overload if only 1 rank
        # we only overload particles in one layer of the domain decomposition
        # so we cannot overload to more than the extent of each partition
        assert overload_length < partition.extent[i] * box_size

    nranks = partition.nranks
    if nranks == 1:
        return data

    rank = partition.rank
    comm = partition.comm
    dimensions = partition.dimensions
    origin = box_size * np.array(partition.origin)
    extent = box_size * np.array(partition.extent)

    neighbors = partition.neighbors

    # Find all overload regions each particle should be in
    overload_left = {}
    overload_right = {}
    for i, x in enumerate(coord_keys):
        _i = np.zeros_like(data[x], dtype=np.int8)
        _i[data[x] < origin[i] + overload_length] = -1

        if structure_key is not None:
            # find all structures present in objects to be overloaded left
            all_structs = np.unique(data[structure_key][_i == -1])
            all_structs = np.setdiff1d(all_structs, -1)
            # add objects with these structure flags to the mask
            all_structs_mask = np.isin(data[structure_key], all_structs)
            _i[all_structs_mask] = -1

        overload_left[i] = _i

        _i = np.zeros_like(data[x], dtype=np.int8)
        _i[data[x] > origin[i] + extent[i] - overload_length] = 1

        if structure_key is not None:
            # find all structures present in objects to be overloaded right
            all_structs = np.unique(data[structure_key][_i == 1])
            # all_structs = np.unique(all_structs, -1)
            # add objects with these structure flags to the mask
            all_structs_mask = np.isin(data[structure_key], all_structs)
            _i[all_structs_mask] = 1

        overload_right[i] = _i

    # Get particle indices of each of the 27 neighbors overload
    exchange_indices = [np.empty(0, dtype=np.int64) for i in range(nranks)]

    def add_exchange_indices(mask: npt.NDArray[np.bool_], idx: tuple[int, ...]) -> None:
        assert len(idx) == dimensions
        n = neighbors[tuple(_i + 1 for _i in idx)]
        if n != rank:
            exchange_indices[n] = np.union1d(exchange_indices[n], np.nonzero(mask)[0])

    corners = itertools.product([0, -1, 1], repeat=partition.dimensions)
    # skip first: will be [0,0,0]
    next(corners)

    for corner in corners:
        mask = np.ones_like(overload_left[0], dtype=np.bool_)
        for d in range(partition.dimensions):
            if corner[d] == 0:
                continue
            mask &= (overload_left[d] == corner[d]) | (overload_right[d] == corner[d])
        add_exchange_indices(mask, corner)

    # Check how many elements will be sent
    send_counts = np.array([len(i) for i in exchange_indices], dtype=np.int32)
    send_idx = np.concatenate(exchange_indices)
    send_displacements = np.insert(np.cumsum(send_counts)[:-1], 0, 0)
    total_to_send = np.sum(send_counts)

    # Check how many elements will be received
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    total_to_receive = np.sum(recv_counts)

    # debug message
    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Overload Debug Rank {i}")
                print(f" - rank sends    {total_to_send:10d} particles")
                print(f" - rank receives {total_to_receive:10d} particles")
                print(f" - send_counts:        {send_counts}")
                print(f" - send_displacements: {send_displacements}")
                print(f" - recv_counts:        {recv_counts}")
                print(f" - recv_displacements: {recv_displacements}")
                for i, x in enumerate(coord_keys):
                    print(f" - overload_left_{x}: {overload_left[i]}")
                    print(f" - overload_right_{x}: {overload_right[i]}")
                print(f" - send_idx: {send_idx}")
                print("", flush=True)
            comm.Barrier()

    # send data all-to-all, each array individually
    data_new = {}
    keys = list(data.keys())
    keys_0 = partition.comm.bcast(keys, root=0)
    assert len(keys) == len(keys_0), "Keys must be the same on all ranks"
    assert all(k in keys_0 for k in keys), "Keys must be the same on all ranks"
    for k in keys_0:
        # prepare send-array
        ds = data[k][send_idx]
        # prepare recv-array
        dr = np.empty(total_to_receive, dtype=ds.dtype)
        # exchange data
        s_msg = [ds, (send_counts, send_displacements), ds.dtype.char]
        r_msg = [dr, (recv_counts, recv_displacements), ds.dtype.char]
        comm.Alltoallv(s_msg, r_msg)
        # add received data to original data
        data_new[k] = np.concatenate((data[k], dr))

    return data_new
