from typing import Mapping, List, Union
import numpy as np
import sys

from .s2_partition import S2Partition
from mpi4py import MPI

ParticleDataT = Mapping[str, np.ndarray]


def s2_distribute(
    partition: S2Partition,
    data: ParticleDataT,
    *,
    theta_key: str = "theta",
    phi_key: str = "phi",
    verbose: Union[bool, int] = False,
    verify_count: bool = True,
    validate_home: bool = False,
) -> ParticleDataT:
    """Distribute particles among MPI ranks according to the S2 partition.

    Parameters
    ----------
    partition:
        The S2 partition to use for the distribution.

    data:
        The particle data to distribute, as a collection of 1-dimensional arrays.
        Each array must have the same length (number of particles) and the map needs
        to contain at least the keys `theta_key` and `phi_key`.

    theta_key:
        The key in `data` that contains the particle theta coordinates (latitude),
        in the range [0, pi].

    phi_key:
        The key in `data` that contains the particle phi coordinates (longitude),
        in the range [0, 2*pi].

    verbose:
        If True, print summary statistics of the distribute. If > 1, print
        statistics of each rank (i.e. how much data each rank sends to every
        other rank).

    verify_count:
        If True, make sure that total number of objects is conserved.

    validate_home:
        If True, validate that each rank indeed owns the particles that it was sent.

    Returns
    -------
    data: ParticleDataT
        The distributed particle data (i.e. the data that this rank owns)

    """
    # count number of particles we have
    total_to_send = len(data[theta_key])

    # verify data is normalized
    assert np.all(data[theta_key] >= 0)
    assert np.all(data[theta_key] < np.pi)
    assert np.all(data[phi_key] >= 0)
    assert np.all(data[phi_key] < 2 * np.pi)

    # ring idx: 0=cap, 1=first ring, 2=second ring, etc.
    if partition.equal_area:
        ring_idx = np.digitize(data[theta_key], partition.ring_thetas)
    else:  # equal theta
        if partition.nranks == 2:
            ring_idx = (data[theta_key] > partition.theta_cap).astype(np.int32)
        else:
            assert partition.ring_dtheta is not None
            ring_idx = (
                (data[theta_key] - partition.theta_cap) // partition.ring_dtheta
            ).astype(np.int32) + 1
            ring_idx = np.clip(ring_idx, 0, len(partition.ring_segments) + 1)

    phi_idx = np.zeros_like(ring_idx, dtype=np.int32)
    mask_is_on_ring = (ring_idx > 0) & (ring_idx <= len(partition.ring_segments))
    phi_idx[mask_is_on_ring] = (
        data[phi_key][mask_is_on_ring]
        / (2 * np.pi)
        * partition.ring_segments[ring_idx[mask_is_on_ring] - 1]
    ).astype(np.int32)

    # rank index where each ring starts
    ring_start_idx = np.zeros(len(partition.ring_segments) + 2, dtype=np.int32)
    ring_start_idx[1] = 1
    ring_start_idx[2:] = np.cumsum(partition.ring_segments) + 1

    # rank index of each particle
    home_idx = ring_start_idx[ring_idx] + phi_idx

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
            [len(data[theta_key]), len(data_new[theta_key])], dtype=np.int64
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
        assert np.all(data_new[theta_key] >= partition.theta_extent[0])
        assert np.all(data_new[theta_key] < partition.theta_extent[1])
        assert np.all(data_new[phi_key] >= partition.phi_extent[0])
        assert np.all(data_new[phi_key] < partition.phi_extent[1])

    return data_new
