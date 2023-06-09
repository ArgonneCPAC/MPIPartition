#!/usr/bin/env python

"""Tests for `mpipartition` package."""

import numpy as np
import pytest

from mpipartition import S2Partition
from mpipartition.spherical_partition.s2_partition import _cap_area


def _test_segmentation(theta_cap, ring_thetas, ring_segments, segment_list):
    assert len(segment_list) == np.sum(ring_segments) + 2
    assert np.isclose(segment_list[0].theta_range[0], 0.0)
    assert np.isclose(segment_list[0].theta_range[1], theta_cap)
    assert np.isclose(segment_list[-1].theta_range[0], np.pi - theta_cap)
    assert np.isclose(segment_list[-1].theta_range[1], np.pi)
    assert np.isclose(segment_list[0].area, _cap_area(theta_cap))
    assert np.isclose(segment_list[-1].area, _cap_area(theta_cap))
    for i in range(len(ring_segments)):
        for j in range(ring_segments[i]):
            idx = np.sum(ring_segments[:i]) + j + 1
            assert np.isclose(segment_list[idx].theta_range[0], ring_thetas[i])
            assert np.isclose(segment_list[idx].theta_range[1], ring_thetas[i + 1])
            assert np.isclose(
                segment_list[idx].area,
                (_cap_area(ring_thetas[i + 1]) - _cap_area(ring_thetas[i]))
                / ring_segments[i],
            )
            assert np.isclose(
                segment_list[idx].phi_range[0],
                2 * np.pi * j / ring_segments[i],
            )
            assert np.isclose(
                segment_list[idx].phi_range[1],
                2 * np.pi * (j + 1) / ring_segments[i],
            )


def _partition(equal_area: bool):
    partition = S2Partition(equal_area=equal_area)
    assert partition.equal_area == equal_area
    assert len(partition.all_s2_segments) == partition.nranks

    # check ring_thetas and ring_segments
    assert np.array(partition.ring_thetas).ndim == 1
    assert np.array(partition.ring_segments).ndim == 1
    assert len(partition.ring_thetas) == len(partition.ring_segments) + 1
    if partition.ring_segments.size:
        assert partition.ring_segments.min() > 0

    # check theta_cap
    assert partition.theta_cap > 0.0
    assert partition.theta_cap <= np.pi / 2.0
    assert np.all(np.diff(partition.ring_thetas) > 0.0)
    assert partition.ring_thetas[0] == partition.theta_cap
    assert np.isclose(partition.ring_thetas[-1], np.pi - partition.theta_cap)

    # check areas
    areas = np.array([r.area for r in partition.all_s2_segments])
    assert np.all(areas > 0.0)
    assert np.all(areas <= 4.0 * np.pi)
    assert np.isclose(np.sum(areas), 4.0 * np.pi)

    # check full segmentation
    _test_segmentation(
        partition.theta_cap,
        partition.ring_thetas,
        partition.ring_segments,
        partition.all_s2_segments,
    )
    return partition


@pytest.mark.mpi
def test_equal_area_s2partition():
    partition = _partition(equal_area=True)
    areas = np.array([r.area for r in partition.all_s2_segments])
    assert np.all(np.isclose(areas, areas[0]))
    assert partition.ring_dtheta is None


@pytest.mark.mpi
def test_equal_dtheta_s2partition():
    partition = _partition(equal_area=False)
    assert partition.ring_dtheta is not None
    if partition.ring_segments.size:
        assert np.all(np.isclose(np.diff(partition.ring_thetas), partition.ring_dtheta))
