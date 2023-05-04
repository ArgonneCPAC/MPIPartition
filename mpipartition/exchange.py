import sys
from typing import Callable, Mapping, Union

import numpy as np

from .partition import Partition

ParticleDataT = Mapping[str, np.ndarray]


def exchange(
    partition: Partition,
    data: dict,
    key: str,
    local_keys: np.ndarray,
    *,
    verbose: bool = False,
    filter_key: Union[int, Callable[[np.ndarray], np.ndarray]] = None,
    do_all2all: bool = False,
    replace_notfound_key: int = None,
):
    """Distribute data among neighboring ranks and all2all by a key

    This function will assign data to the rank that owns the key. The keys that the
    local rank owns are given by ``local_keys``, which should be unique. The keys of the
    data that the local rank currently has is in ``data[key]``. Certain values can be
    ignored by setting filter_key to that value or by setting filter_key to a
    (vectorized) function that returns ``True`` for keys that should be redistributed
    and ``False`` for keys that should be ignored.

    Parameters
    ----------


    Returns
    -------
    """
    comm = partition.comm
    rank = partition.rank
    nranks = partition.nranks
    if nranks == 1:
        return data

    if do_all2all:
        # exchange particles with all ranks
        exchange_comm = comm
        exchange_nranks = nranks
        exchange_Alltoall = exchange_comm.Alltoall
        exchange_Alltoallv = exchange_comm.Alltoallv
        exchange_Allgather = exchange_comm.Allgather
        exchange_Allgatherv = exchange_comm.Allgatherv

    else:
        # exchange particles with the neighboring ranks
        exchange_comm = partition.comm_neighbor
        exchange_nranks = partition.comm_neighbor.Get_size()
        exchange_Alltoall = exchange_comm.Neighbor_alltoall
        exchange_Alltoallv = exchange_comm.Neighbor_alltoallv
        exchange_Allgather = exchange_comm.Neighbor_allgather
        exchange_Allgatherv = exchange_comm.Neighbor_allgatherv

    localcount = len(data[key])
    data_keys = np.unique(data[key])
    if filter_key is not None:
        if callable(filter_key):
            data_keys = data_keys[filter_key(data_keys)]
        else:
            data_keys = data_keys[data_keys != filter_key]

    # find local matches
    islocal = np.isin(data_keys, local_keys, assume_unique=True)
    nonlocal_data = data_keys[~islocal]

    # communicate nonlocal descendants
    local_orphan_count = np.array([len(nonlocal_data)], dtype=np.int32)
    orphan_counts = np.empty(exchange_nranks, dtype=np.int32)
    exchange_Allgather(local_orphan_count, orphan_counts)
    total_orphan_count = np.sum(orphan_counts)
    orphan_offsets = np.insert(np.cumsum(orphan_counts)[:-1], 0, 0)
    orphan_data = np.empty(total_orphan_count, dtype=nonlocal_data.dtype)
    orphan_ranks = np.empty(total_orphan_count, dtype=np.int32)
    for i in range(exchange_nranks):
        low = orphan_offsets[i]
        high = low + orphan_counts[i]
        orphan_ranks[low:high] = i
    exchange_Allgatherv(
        nonlocal_data,
        [orphan_data, (orphan_counts, orphan_offsets), nonlocal_data.dtype.char],
    )

    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Debug Desc Exchange (nonlocal), rank {i}")
                print(f" - send {local_orphan_count}")
                print(f" - recv {orphan_counts}")
                print(f"", flush=True)
            comm.Barrier()

    # check if we have any of them
    orphan_islocal = np.isin(orphan_data, local_keys)

    # ask
    orphan_requests_send = orphan_data[orphan_islocal]
    orphan_requests_send_ranks = orphan_ranks[orphan_islocal]
    orphan_requests_send_offsets = np.searchsorted(
        orphan_requests_send_ranks, np.arange(exchange_nranks)
    )
    orphan_requests_send_counts = (
        np.append(orphan_requests_send_offsets[1:], len(orphan_requests_send))
        - orphan_requests_send_offsets
    )
    orphan_requests_recv_counts = np.empty_like(orphan_requests_send_counts)
    exchange_Alltoall(orphan_requests_send_counts, orphan_requests_recv_counts)
    orphan_requests_recv_total = np.sum(orphan_requests_recv_counts)
    orphan_requests_recv_offsets = np.insert(
        np.cumsum(orphan_requests_recv_counts)[:-1], 0, 0
    )
    orphan_requests_recv = np.empty(
        orphan_requests_recv_total, dtype=orphan_requests_send.dtype
    )
    exchange_Alltoallv(
        [
            orphan_requests_send,
            (orphan_requests_send_counts, orphan_requests_send_offsets),
            orphan_requests_send.dtype.char,
        ],
        [
            orphan_requests_recv,
            (orphan_requests_recv_counts, orphan_requests_recv_offsets),
            orphan_requests_recv.dtype.char,
        ],
    )

    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Debug Desc Exchange (request), rank {i}")
                print(f" - will request {np.sum(orphan_islocal)} in total")
                print(f" - send req {orphan_requests_send_counts}")
                print(f" - recv req {orphan_requests_recv_counts}")
                print(f"", flush=True)
            comm.Barrier()

    # verify that we don't aks ourselves for particles
    if do_all2all and (
        orphan_requests_send_counts[rank] != 0 or orphan_requests_recv_counts[rank] != 0
    ):
        print(
            f"Error in exchange: rank {rank} is asking itself for an orphan halo: "
            f"{orphan_requests_send_counts[rank]}/{orphan_requests_recv_counts[rank]}",
            file=sys.stderr,
            flush=True,
        )
        comm.Abort()

    # prepare data to send
    orphan_requests_indices = []
    orphan_requests_mask = np.zeros(localcount, dtype=np.bool)
    for i in range(exchange_nranks):
        req = orphan_requests_recv[
            orphan_requests_recv_offsets[i] : orphan_requests_recv_offsets[i]
            + orphan_requests_recv_counts[i]
        ]
        mask = np.isin(data[key], req)
        orphan_requests_indices.append(np.nonzero(mask)[0])
        orphan_requests_mask |= mask
    orphan_requests_send_counts = np.array(
        [len(i) for i in orphan_requests_indices], dtype=np.int32
    )
    orphan_requests_recv_counts = np.empty_like(orphan_requests_send_counts)
    exchange_Alltoall(orphan_requests_send_counts, orphan_requests_recv_counts)
    orphan_requests_recv_total = np.sum(orphan_requests_recv_counts)
    orphan_requests_send_offsets = np.insert(
        np.cumsum(orphan_requests_send_counts)[:-1], 0, 0
    )
    orphan_requests_recv_offsets = np.insert(
        np.cumsum(orphan_requests_recv_counts)[:-1], 0, 0
    )
    orphan_requests_indices = np.concatenate(orphan_requests_indices)

    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Debug Desc Exchange (to exchange), rank {i}")
                print(f" - send {orphan_requests_send_counts}")
                print(f" - recv {orphan_requests_recv_counts}")
                print(f"", flush=True)
            comm.Barrier()

    data_new = {}
    for k in data.keys():
        orphan_requests_send = data[k][orphan_requests_indices]
        orphan_requests_recv = np.empty(orphan_requests_recv_total, dtype=data[k].dtype)
        exchange_Alltoallv(
            [
                orphan_requests_send,
                (orphan_requests_send_counts, orphan_requests_send_offsets),
                orphan_requests_send.dtype.char,
            ],
            [
                orphan_requests_recv,
                (orphan_requests_recv_counts, orphan_requests_recv_offsets),
                orphan_requests_recv.dtype.char,
            ],
        )
        data_new[k] = np.concatenate(
            (data[k][~orphan_requests_mask], orphan_requests_recv)
        )

    if verbose > 1 and rank == 0:
        print("Exchange succeeded, verifying data integrity", flush=True)
    # comm.Barrier()

    # Verification
    localcount_after = len(data_new[key])
    localcount_missmatch = local_orphan_count[0]
    # calculate new missmatch
    my_data = np.unique(data_new[key])
    my_data = my_data[my_data >= 0]
    islocal = np.isin(my_data, local_keys, assume_unique=True)
    missing_keys = my_data[~islocal]
    localcount_missmatch_after = len(missing_keys)
    localcounts = np.array(
        [
            localcount,
            localcount_after,
            localcount_missmatch,
            localcount_missmatch_after,
        ],
        dtype=np.int64,
    )
    totalcounts = np.empty_like(localcounts)
    comm.Allreduce(localcounts, totalcounts)
    (
        totalcount_before,
        totalcount_after,
        totalcount_missmatch,
        totalcount_missmatch_after,
    ) = totalcounts

    if verbose and rank == 0:
        print(f"exchange summary ({'all2all' if do_all2all else 'neighbors'}):")
        print(
            f"   Ntot -> Ntot: {totalcount_before:10d} -> {totalcount_after:10d} "
            "(should remain the same)"
        )
        print(
            f"   Orph -> Orph: {totalcount_missmatch:10d} -> "
            f"{totalcount_missmatch_after:10d} (should be 0 after)"
        )
        print("", flush=True)

    # did we conserve number of particles?
    if rank == 0 and totalcount_before != totalcount_after:
        print(
            "Error in exchange: Lost halos during progenitor exchange: "
            f"{totalcount_before} -> {totalcount_after}",
            file=sys.stderr,
            flush=True,
        )
        comm.Abort()

    # if we were not able to assign all orphans to the neighbors, try all2all
    if not do_all2all and totalcount_missmatch_after > 0:
        if verbose and rank == 0:
            print(
                "exchange all2all since neighbor exchange was not able to assign all: "
                f"{totalcount_missmatch} -> {totalcount_missmatch_after}",
                flush=True,
            )
        return exchange(
            partition,
            data_new,
            key,
            local_keys,
            verbose=verbose,
            filter_key=filter_key,
            do_all2all=True,
            replace_notfound_key=replace_notfound_key,
        )

    # if we are still not able to assign all orphans, replace key or abort after
    # printing some debug messages
    if replace_notfound_key is not None and localcount_missmatch_after > 0:
        d = data_new[key]
        d[np.isin(d, missing_keys)] = replace_notfound_key
    for i in range(nranks):
        if rank == i and localcount_missmatch_after != 0:
            print(
                f"Warning from rank {rank} in exchange: Unable to assign all "
                f"progenitors to correct ranks (failed for "
                f"{localcount_missmatch_after} out of {localcount_missmatch})"
            )
            print("Could not assign keys: ", missing_keys)
            print("", flush=True)
        comm.Barrier()

    if rank == 0 and totalcount_missmatch_after != 0:
        if replace_notfound_key is None:
            print(
                f"Error in exchange: Unable to assign all progenitors to correct ranks "
                f"(tried to reassign {totalcount_missmatch}, failed for "
                f"{totalcount_missmatch_after})",
                file=sys.stderr,
                flush=True,
            )
            comm.Abort()
        else:
            print(
                f"Warning in exchange: Unable to assign all progenitors to correct "
                f"ranks (tried to reassign {totalcount_missmatch}, failed for "
                f"{totalcount_missmatch_after}), replacing missing values with "
                f"{replace_notfound_key}",
                flush=True,
            )

    return data_new
