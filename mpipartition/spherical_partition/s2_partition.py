import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, List
from dataclasses import dataclass

from mpi4py import MPI


def _cap_area(theta):
    h = 1 - np.cos(theta)
    return 2 * np.pi * h


def _cap_angle(area):
    h = area / (2 * np.pi)
    return np.arccos(1 - h)


def _s2_partition(n, adjust_theta=True):
    # Following the algorithm by Paul Leopardi (https://eqsp.sourceforge.net/)
    # TODO: generalize to arbitrary number of dimensions
    assert n > 1
    if n == 2:
        return (
            np.pi / 2,
            np.array([np.pi / 2], dtype=np.float64),
            np.array([], dtype=np.int32),
        )

    target_area = 4 * np.pi / n

    # 1: polar caps
    theta_c = _cap_angle(target_area)

    # 2-4: rings
    delta_i = np.sqrt(target_area)  # ideal spacing
    n_rings_i = (np.pi - 2 * theta_c) / delta_i
    n_rings = max(int(n_rings_i + 0.5), 1)
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
        for j in range(ring_segments[i]):
            segments.append(
                S2Segment(
                    (theta_start, theta_end),
                    (phi_edges[j], phi_edges[j + 1]),
                    area,
                    edge_length,
                )
            )

    # cap
    segments.append(
        S2Segment((np.pi - theta_cap, np.pi), (0, 2 * np.pi), cap_area, cap_edge_length)
    )

    return segments


def _print_segmentation_info(
    nranks, theta_cap, ring_thetas, ring_segments, precision=3
):
    print(f"Segmentation statistics for {nranks} ranks:")
    print(f"  polar cap angle: {theta_cap:.{precision}f}")
    print(f"  number of rings: {len(ring_segments)}")
    for i in range(len(ring_segments)):
        print(
            f"    ring {i:3d}: {ring_segments[i]:3d} segments between "
            f"theta=[{ring_thetas[i]:.{precision}f}, {ring_thetas[i+1]:.{precision}f}]]"
        )


# area imbalance
def _print_area_imabalance(segments: List[S2Segment], precision=3):
    areas = np.array([r.area for r in segments])
    assert np.isclose(np.sum(areas), 4 * np.pi)
    print("  Segment area imbalance:")
    print(f"    max/min: {np.max(areas) / np.min(areas):.{precision}f}")
    print(f"    max/avg: {np.max(areas) / np.mean(areas):.{precision}f}")


def _print_edge_to_area_ratio(segments: List[S2Segment], precision=3):
    areas = np.array([r.area for r in segments])
    edge_lengths = np.array([r.edge_length for r in segments])
    total_ratio = np.sum(edge_lengths) / np.sum(areas)
    print(f"  Total edge/area ratio: {total_ratio:.{precision}f}")


class S2Partition:
    """An MPI decomposition of the spherical shell into equal-area segments

    Parameters
    ----------
    equal_area : bool
        If True, the spherical shell is divided into equal-area segments. If False, use
        equally spaced rings (in theta) instead.

    comm : MPI.Comm
        The MPI communicator to use for the decomposition (default: COMM_WORLD)

    verbose : bool
        If True, rank 0 will print information about the segmentation.

    """

    # parition properties
    _comm: MPI.Comm
    _nranks: int
    _theta_cap: float
    _ring_thetas: npt.NDArray[np.float64]
    _ring_segments: npt.NDArray[np.int64]
    _equal_area: bool
    _ring_dtheta: Optional[float]
    _all_s2_segments: List[S2Segment]

    # rank properties
    _rank: int
    _s2_segment: S2Segment

    @property
    def comm(self):
        """MPI Communicator"""
        return self._comm

    @property
    def rank(self) -> int:
        """the MPI rank of this processor"""
        return self._rank

    @property
    def nranks(self) -> int:
        """the total number of processors"""
        return self._nranks

    @property
    def equal_area(self) -> bool:
        """whether the partition is equal-area"""
        return self._equal_area

    @property
    def theta_cap(self) -> float:
        """the polar cap angle"""
        return self._theta_cap

    @property
    def ring_thetas(self) -> npt.NDArray[np.float64]:
        """the theta boundaries of all rings"""
        return self._ring_thetas

    @property
    def ring_segments(self) -> npt.NDArray[np.int64]:
        """the number of segments in each ring"""
        return self._ring_segments

    @property
    def ring_dtheta(self) -> Optional[float]:
        """the theta spacing between rings (only for non-equal-area partitions)"""
        return self._ring_dtheta

    @property
    def phi_extent(self) -> Tuple[float, float]:
        """the phi extent of the segment assigned to this rank"""
        return self._s2_segment.phi_range

    @property
    def theta_extent(self) -> Tuple[float, float]:
        """the theta extent of the segment assigned to this rank"""
        return self._s2_segment.theta_range

    @property
    def area(self) -> float:
        """the area of the segment assigned to this rank"""
        return self._s2_segment.area

    @property
    def all_phi_extents(self) -> npt.NDArray[np.float64]:
        """the phi extent of all segments, shape (nranks, 2)"""
        return np.array([s.phi_range for s in self._all_s2_segments])

    @property
    def all_theta_extents(self) -> npt.NDArray[np.float64]:
        """the theta extent of all segments, shape (nranks, 2)"""
        return np.array([s.theta_range for s in self._all_s2_segments])

    def __init__(
        self,
        *,
        equal_area: bool = True,
        comm: MPI.Comm = MPI.COMM_WORLD,
        verbose: bool = False,
    ):
        self._comm = comm
        self._rank = self._comm.Get_rank()
        self._nranks = self._comm.Get_size()
        self._equal_area = equal_area
        self._theta_cap, self._ring_thetas, self._ring_segments = _s2_partition(
            self._nranks, equal_area
        )
        if not equal_area:
            if len(self._ring_thetas) > 1:
                self._ring_dtheta = self._ring_thetas[1] - self._ring_thetas[0]
            else:
                self._ring_dtheta = 0.0
        else:
            self._ring_dtheta = None

        self._all_s2_segments = _build_s2_segment_list(
            self._theta_cap, self._ring_thetas, self._ring_segments
        )
        assert len(self._all_s2_segments) == self._nranks
        self._s2_segment = self._all_s2_segments[self._rank]

        if verbose and self._rank == 0:
            _print_segmentation_info(
                self._nranks,
                self._theta_cap,
                self._ring_thetas,
                self._ring_segments,
            )
            _print_area_imabalance(self._all_s2_segments)
            _print_edge_to_area_ratio(self._all_s2_segments)
            print()


def visualize_s2_partition(
    nranks: int, equal_area: bool = True, use_mollweide: bool = True, fig=None
):
    """Visualize the S2 partitioning of the sphere.

    Parameters
    ----------
    nranks : int
        Number of ranks to partition the sphere into.
    equal_area : bool, optional
        If True, partition the sphere into equal area regions by adjusting theta.
        Otherwise, keep delta_theta of rings constant.
    use_mollweide : bool, optional
        If True, use the Mollweide projection. Otherwise, use a regular plot.
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, create a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")

    if fig is None:
        fig = plt.figure()
    if use_mollweide:
        ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    else:
        ax = fig.add_subplot(1, 1, 1)

    theta_c, theta_f, n_segments = _s2_partition(nranks, equal_area)

    for i, nsegments in enumerate(n_segments):
        theta_lo = theta_f[i]
        theta_hi = theta_f[i + 1]
        if use_mollweide:
            theta_lo -= np.pi / 2
            theta_hi -= np.pi / 2

        # draw vertical bars
        for j in range(nsegments):
            ax.plot(
                (2 * np.pi / nsegments * j - np.pi) * np.ones(2),
                [theta_lo, theta_hi],
                color="black",
                linewidth=0.5,
            )
        # draw upper ring
        ax.axhline(theta_lo, color="black", linewidth=0.5)
    # draw final ring (only if we have rings!)
    if n_segments.size:
        ax.axhline(theta_hi, color="black", linewidth=0.5)
    else:
        ax.axhline(theta_c, color="black", linewidth=0.5)

    if not use_mollweide:
        ax.set(
            xlabel=r"$\phi$",
            ylabel=r"$\theta$",
            xlim=(-np.pi, np.pi),
            ylim=(0, np.pi),
        )

    return fig, ax
