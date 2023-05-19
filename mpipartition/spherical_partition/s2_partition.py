import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, List
from dataclasses import dataclass

from mpi4py import MPI

_comm = MPI.COMM_WORLD
_rank = _comm.Get_rank()
_nranks = _comm.Get_size()


def _cap_area(theta):
    h = 1 - np.cos(theta)
    return 2 * np.pi * h


def _cap_angle(area):
    h = area / (2 * np.pi)
    return np.arccos(1 - h)


def _s2_partition(n, adjust_theta=True, verbose=False):
    # Following the algorithm by Paul Leopardi (https://eqsp.sourceforge.net/)
    # TODO: generalize to arbitrary number of dimensions
    assert n > 2

    target_area = 4 * np.pi / n

    # 1: polar caps
    theta_c = _cap_angle(target_area)

    # 2-4: rings
    delta_i = np.sqrt(target_area)  # ideal spacing
    n_rings_i = (np.pi - 2 * theta_c) / delta_i
    n_rings = int(n_rings_i + 0.5)

    # 5-6: ring segments
    delta_f = (np.pi - 2 * theta_c) / n_rings
    theta_f = theta_c + np.arange(n_rings) * delta_f
    theta_f = np.append(theta_f, np.pi - theta_c)

    n_segments_i = (_cap_area(theta_f[1:]) - _cap_area(theta_f[:-1])) / target_area
    n_segments = np.zeros(n_rings, dtype=np.int32)
    remainder = 0
    for i in range(n_rings):
        ni = int(n_segments_i[i] + remainder + 0.5)
        remainder += n_segments_i[i] - ni
        n_segments[i] = ni

    assert abs(remainder) < 1e-6
    assert np.sum(n_segments) + 2 == n

    # 7: adjust theta_f for equal area
    if adjust_theta:
        areas = target_area * np.array(n_segments)
        cum_areas = target_area + np.cumsum(areas)
        theta_f[1:] = _cap_angle(cum_areas)

    if verbose:
        print(f"Segmentation statistics for {n} ranks:")
        print(f"  polar cap angle: {theta_c:.3f}")
        print(f"  number of rings: {n_rings}")
        for i in range(n_rings):
            print(
                f"    ring {i:3d}: {n_segments[i]:3d} segments "
                f"between theta=[{theta_f[i]:.3f}, {theta_f[i+1]:.3f}]]"
            )

    return theta_c, theta_f, n_segments


@dataclass
class S2Segment:
    theta_range: Tuple[float]
    phi_range: Tuple[float]
    area: float
    edge_length: float


def _build_s2_segment_list(theta_cap, ring_thetas, ring_segments):
    segments = []

    # cap
    cap_area = _cap_area(theta_cap)
    cap_edge_length = 2 * np.pi * np.sin(theta_cap)
    segments.append(
        S2Segment((0, theta_cap), (0, 2 * np.pi), cap_area, cap_edge_length)
    )

    # rings
    for i in range(len(ring_segments)):
        theta_start = ring_thetas[i]
        theta_end = ring_thetas[i + 1]
        area = (_cap_area(theta_end) - _cap_area(theta_start)) / ring_segments[i]
        phi_edges = np.linspace(0, 2 * np.pi, ring_segments[i] + 1, endpoint=True)
        edge_length_top = 2 * np.pi * np.sin(theta_start) / ring_segments[i]
        edge_length_bottom = 2 * np.pi * np.sin(theta_end) / ring_segments[i]
        edge_length = (
            edge_length_top + edge_length_bottom + 2 * (theta_end - theta_start)
        )
        segments.append(
            S2Segment((theta_start, theta_end), phi_edges, area, edge_length)
        )

    # cap
    segments.append(
        S2Segment((np.pi - theta_cap, np.pi), (0, 2 * np.pi), cap_area, cap_edge_length)
    )

    return segments


# area imbalance
def _print_area_imabalance(segments: List[S2Segment]):
    areas = np.array([r.area for r in segments])
    assert np.isclose(np.sum(areas), 4 * np.pi)

    print(f"area imbalance max/min: {np.max(areas) / np.min(areas):.3f}")
    print(f"area imbalance max/avg: {np.max(areas) / np.mean(areas):.3f}")


def _print_edge_to_area_ratio(segments: List[S2Segment]):
    areas = np.array([r.area for r in segments])
    edge_lengths = np.array([r.edge_length for r in segments])
    print(f"total edge/area ratio: {np.sum(edge_lengths) / np.sum(areas):.3f}")


class S2Partition:
    # parition properties
    nranks: int
    theta_cap: float
    ring_thetas: npt.NDArray[np.float64]
    ring_segments: npt.NDArray[np.int64]
    equal_area: bool
    ring_dtheta: Optional[float]
    all_s2_segments: List[S2Segment]

    # rank properties
    rank: int
    s2_segment: S2Segment

    def __init__(self, equal_area: bool = True, verbose: bool = False):
        self.rank = _rank
        self.nranks = _nranks
        self.equal_area = equal_area
        self.theta_cap, self.ring_thetas, self.ring_segments = _s2_partition(
            self.nranks, equal_area, verbose=verbose and self.rank == 0
        )
        if not equal_area:
            self.ring_dtheta = self.ring_thetas[1] - self.ring_thetas[0]
        else:
            self.ring_dtheta = None

        self.all_s2_segments = _build_s2_segment_list(
            self.theta_cap, self.ring_thetas, self.ring_segments
        )
        assert len(self.all_s2_segments) == self.nranks
        self.s2_segment = self.all_s2_segments[self.rank]

        if verbose and self.rank == 0:
            print(f"\nBalance statistics for {self._nranks} ranks")
            _print_area_imabalance(self.all_s2_segments)
            _print_edge_to_area_ratio(self.all_s2_segments)
            print()
