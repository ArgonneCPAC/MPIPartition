from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from mpi4py import MPI
import sys

if TYPE_CHECKING:
    from .partition import Partition
    from .spherical_partition import S2Partition

ParticleDataT = dict[str, np.ndarray]


def distribute_dataset_by_home(
    partition: Partition | S2Partition,
    data: ParticleDataT,
    home_idx: np.ndarray,
    *,
    verbose: int = 0,
    verify_count: bool = True,
) -> ParticleDataT:
    total_to_send = len(home_idx)
    for d in data.values():
        assert len(d) == total_to_send, "All data arrays must have the same length"

    # sort by rank
    s = np.argsort(home_idx)
    home_idx = home_idx[s]

    # offsets and counts
    send_displacements = np.searchsorted(home_idx, np.arange(partition.nranks))
    send_displacements = send_displacements.astype(np.int32)
    send_counts = np.append(send_displacements[1:], total_to_send) - send_displacements
    send_counts = send_counts.astype(np.int32)

    # announce to each rank how many objects will be sent
    recv_counts = np.empty_like(send_counts)
    partition.comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)

    # number of objects that this rank will receive
    total_to_receive = np.sum(recv_counts)

    # debug message
    if verbose > 1:
        for i in range(partition.nranks):
            if partition.rank == i:
                print(f"Distribute Debug Rank {i}")
                print(f" - rank has {total_to_send} particles")
                print(f" - rank receives {total_to_receive} particles")
                print(f" - send_counts:        {send_counts}")
                print(f" - send_displacements: {send_displacements}")
                print(f" - recv_counts:        {recv_counts}")
                print(f" - recv_displacements: {recv_displacements}")
                print(f"", flush=True)
            partition.comm.Barrier()

    # send data all-to-all, each array individually
    data_new = {k: np.empty(total_to_receive, dtype=data[k].dtype) for k in data.keys()}

    for k in data.keys():
        d = data[k][s]
        s_msg = [d, (send_counts, send_displacements), d.dtype.char]
        r_msg = [data_new[k], (recv_counts, recv_displacements), d.dtype.char]
        partition.comm.Alltoallv(s_msg, r_msg)

    if verify_count:
        key0 = list(data.keys())[0]
        local_counts = np.array([len(data[key0]), len(data_new[key0])], dtype=np.int64)
        global_counts = np.empty_like(local_counts)
        partition.comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
        if partition.rank == 0 and global_counts[0] != global_counts[1]:
            print(
                f"Error in distribute: particle count during distribute was not "
                f"maintained ({global_counts[0]} -> {global_counts[1]})",
                file=sys.stderr,
                flush=True,
            )
            partition.comm.Abort()

    return data_new
