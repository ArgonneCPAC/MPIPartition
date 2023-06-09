from typing import List, Mapping, Union

import numpy as np
import numba

from .s2_partition import S2Partition

ParticleDataT = Mapping[str, np.ndarray]


@numba.jit(nopython=True)
def _count_neighbors(
    theta: np.ndarray,
    phi: np.ndarray,
    overload_angle: float,
    my_rank: int,
    segment_theta_low: np.ndarray,
    segment_theta_high: np.ndarray,
    segment_phi_low: np.ndarray,
    segment_phi_high: np.ndarray,
    send_counts: np.ndarray,
    send_counts_by_rank: np.ndarray,
):
    npart = len(theta)
    nsegments = len(segment_theta_low)
    for i in range(npart):
        for j in range(nsegments):
            if j == my_rank:
                continue
            phi_i = phi[i]
            if phi_i < segment_phi_low[j] - overload_angle:
                phi_i += 2 * np.pi
            if phi_i >= segment_phi_high[j] + overload_angle:
                phi_i -= 2 * np.pi
            if (
                theta[i] >= segment_theta_low[j] - overload_angle
                and theta[i] < segment_theta_high[j] + overload_angle
                and phi_i >= segment_phi_low[j] - overload_angle
                and phi_i < segment_phi_high[j] + overload_angle
            ):
                send_counts[i] += 1
                send_counts_by_rank[j] += 1


@numba.jit(nopython=True)
def _calculate_partition(
    theta: np.ndarray,
    phi: np.ndarray,
    overload_angle: float,
    my_rank: int,
    segment_theta_low: np.ndarray,
    segment_theta_high: np.ndarray,
    segment_phi_low: np.ndarray,
    segment_phi_high: np.ndarray,
    send_offset_by_rank: np.ndarray,
    send_permutation: np.ndarray,
):
    send_count_by_rank = np.zeros_like(send_offset_by_rank)
    npart = len(theta)
    nsegments = len(segment_theta_low)
    for i in range(npart):
        for j in range(nsegments):
            if j == my_rank:
                continue
            phi_i = phi[i]
            if phi_i < segment_phi_low[j] - overload_angle:
                phi_i += 2 * np.pi
            if phi_i >= segment_phi_high[j] + overload_angle:
                phi_i -= 2 * np.pi
            if (
                theta[i] >= segment_theta_low[j] - overload_angle
                and theta[i] < segment_theta_high[j] + overload_angle
                and phi_i >= segment_phi_low[j] - overload_angle
                and phi_i < segment_phi_high[j] + overload_angle
            ):
                send_permutation[send_offset_by_rank[j] + send_count_by_rank[j]] = i
                send_count_by_rank[j] += 1


def s2_overload(
    partition: S2Partition,
    data: ParticleDataT,
    overload_angle: float,
    *,
    theta_key: str = "theta",
    phi_key: str = "phi",
    verbose: Union[bool, int] = False,
):
    # verify data is normalized
    assert np.all(data[theta_key] >= 0)
    assert np.all(data[theta_key] <= np.pi)
    assert np.all(data[phi_key] >= 0)
    assert np.all(data[phi_key] <= 2 * np.pi)

    # count for each particle to many ranks it needs to be sent
    send_count_by_particle = np.zeros_like(data[theta_key], dtype=np.int32)
    send_counts = np.zeros(partition.nranks, dtype=np.int32)  # by rank

    segment_theta_low = np.array([s.theta_range[0] for s in partition.all_s2_segments])
    segment_theta_high = np.array([s.theta_range[1] for s in partition.all_s2_segments])
    segment_phi_low = np.array([s.phi_range[0] for s in partition.all_s2_segments])
    segment_phi_high = np.array([s.phi_range[1] for s in partition.all_s2_segments])

    _count_neighbors(
        data[theta_key],
        data[phi_key],
        overload_angle,
        partition.rank,
        segment_theta_low,
        segment_theta_high,
        segment_phi_low,
        segment_phi_high,
        send_count_by_particle,
        send_counts,
    )
    total_send_count = np.sum(send_count_by_particle)
    assert np.sum(send_counts) == total_send_count
    send_displacements = np.insert(np.cumsum(send_counts)[:-1], 0, 0)

    send_permutation = np.empty(total_send_count, dtype=np.int64)
    send_permutation[:] = -1

    _calculate_partition(
        data[theta_key],
        data[phi_key],
        overload_angle,
        partition.rank,
        segment_theta_low,
        segment_theta_high,
        segment_phi_low,
        segment_phi_high,
        send_displacements,
        send_permutation,
    )
    assert np.all(send_permutation >= 0)

    # Check how many elements will be received
    recv_counts = np.empty_like(send_counts)
    partition.comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    total_receive_count = np.sum(recv_counts)

    # debug message
    if verbose > 1:
        for i in range(partition.nranks):
            if partition.rank == i:
                print(f"Overload Debug Rank {i}")
                print(f" - rank sends    {total_send_count:10d} particles")
                print(f" - rank receives {total_receive_count:10d} particles")
                print(f" - send_counts:        {send_counts}")
                print(f" - send_displacements: {send_displacements}")
                print(f" - recv_counts:        {recv_counts}")
                print(f" - recv_displacements: {recv_displacements}")
                print("", flush=True)
            partition.comm.Barrier()

    # send data all-to-all, each array individually
    data_new = {}

    for k in data.keys():
        # prepare send-array
        ds = data[k][send_permutation]
        # prepare recv-array
        dr = np.empty(total_receive_count, dtype=ds.dtype)
        # exchange data
        s_msg = [ds, (send_counts, send_displacements), ds.dtype.char]
        r_msg = [dr, (recv_counts, recv_displacements), ds.dtype.char]
        partition.comm.Alltoallv(s_msg, r_msg)
        # add received data to original data
        data_new[k] = np.concatenate((data[k], dr))

    return data_new
