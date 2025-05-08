from typing import Union
import numpy as np

from .s2_partition import S2Partition
from .._send_home import distribute_dataset_by_home

ParticleDataT = dict[str, np.ndarray]


def s2_distribute(
    partition: S2Partition,
    data: ParticleDataT,
    *,
    theta_key: str = "theta",
    phi_key: str = "phi",
    verbose: Union[bool, int] = False,
    verify_count: bool = True,
    validate_home: bool = False,
    all2all_iterations: int = 1,
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

    all2all_iterations:
        The number of iterations to use for the all-to-all communication.
        This is useful for large datasets, where MPI_Alltoallv may fail.

    Returns
    -------
    data: ParticleDataT
        The distributed particle data (i.e. the data that this rank owns)

    """

    # verify data is normalized
    assert np.all(data[theta_key] >= 0)
    assert np.all(data[theta_key] <= np.pi)
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
            ring_idx[data[theta_key] == np.pi] -= 1  # handle cases where theta == pi

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

    data_new = distribute_dataset_by_home(
        partition,
        data,
        home_idx=home_idx,
        verbose=verbose,
        verify_count=verify_count,
        all2all_iterations=all2all_iterations,
    )

    if validate_home:
        assert np.all(data_new[theta_key] >= partition.theta_extent[0])
        if partition.theta_extent[1] < np.pi:
            assert np.all(data_new[theta_key] < partition.theta_extent[1])
        else:
            # bottom cap, we allow theta == pi
            assert np.all(data_new[theta_key] <= partition.theta_extent[1])
        assert np.all(data_new[phi_key] >= partition.phi_extent[0])
        assert np.all(data_new[phi_key] < partition.phi_extent[1])

    return data_new
