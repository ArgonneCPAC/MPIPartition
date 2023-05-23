from typing import Mapping, List, Union
import numpy as np
import sys

from .s2_partition import S2Partition
from mpi4py import MPI

ParticleDataT = Mapping[str, np.ndarray]


def s2_distribute(
    partition: S2Partition,
    data: ParticleDataT,
    coord_keys: List[str],
    *,
    verbose: Union[bool, int] = False,
    verify_count: bool = True,
    validate_home: bool = False,
) -> ParticleDataT:
    assert len(coord_keys) == 2
    theta = coord_keys[0]
    phi = coord_keys[1]

    # count number of particles we have
    total_to_send = len(data[coord_keys[0]])

    # verify data is normalized
    assert np.all(data[theta] >= 0)
    assert np.all(data[theta] <= np.pi)
    assert np.all(data[phi] >= 0)
    assert np.all(data[phi] <= 2 * np.pi)

    # ring idx: -1=cap, 0=first ring, 1=second ring, etc.
    if partition.equal_area:
        ring_idx = np.digitize(data[theta], partition.ring_thetas, right=True)
    else:  # equal theta
        assert partition.ring_dtheta is not None
        ring_idx = int((data[theta] - partition.theta_cap) // partition.ring_dtheta)

    phi_idx = np.zeros_like(ring_idx, dtype=np.int32)
    mask_ring = (ring_idx >= 0) & (ring_idx < len(partition.ring_segments))
    phi_idx[mask_ring] = (
        phi[mask_ring] / (2 * np.pi) * partition.ring_segments[ring_idx[mask_ring]]
    ).astype(np.int32)

    # add one for the cap
    home_idx = np.sum(partition.ring_segments[:ring_idx], axis=0) + phi_idx + 1

    assert np.all(home_idx >= 0)
    assert np.all(home_idx < partition.nranks)

    # TODO: this code is duplicated in distribute.py – unify!

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
        local_counts = np.array(
            [len(data[coord_keys[0]]), len(data_new[coord_keys[0]])], dtype=np.int64
        )
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

    if validate_home:
        assert np.all(data_new[theta] >= partition.s2_segment.theta_range[0])
        assert np.all(data_new[theta] < partition.s2_segment.theta_range[1])
        assert np.all(data_new[phi] >= partition.s2_segment.phi_range[0])
        assert np.all(data_new[phi] < partition.s2_segment.phi_range[1])

    return data_new
