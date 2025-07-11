"""MPI Partitioning of a cube"""

import itertools
import sys

import numpy as np
from mpi4py import MPI


def _factorize(n: int) -> list[int]:
    i = 2
    factors = []
    while i <= n:
        if (n % i) == 0:
            factors.append(i)
            n //= i
        else:
            i = i + 1
    return factors


def _distribute_factors(
    factors: list[int], target: list[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    current_topo = np.ones_like(target)
    remaining_topo = np.copy(np.array(target))
    for f in factors[::-1]:
        commensurate = (remaining_topo % f) == 0
        if not np.any(commensurate):
            raise RuntimeError(
                "commensurate topology impossible with given rank number and target topo"
            )
        # add to lowest possible number
        s = np.argsort(current_topo)
        idx = s[np.nonzero(commensurate[s])[0][0]]
        current_topo[idx] *= f
        remaining_topo[idx] /= f
    return current_topo, remaining_topo


class Partition:
    """An MPI partition of a cubic volume

    Parameters
    ----------

    dimension: int
        Numer of dimensions of the volume cube. Default: 3

    create_neighbor_topo : boolean
        If `True`, an additional graph communicator will be initialized
        connecting all direct neighbors (3**dimension - 1) symmetrically

    commensurate_topo : List[int]
        A proportional target topology for decomposition. When specified, a partition
        will be created so that `commensurate_topo[i] % partition.decomposition[i] == 0`
        for all `i`. The code will raise a RuntimeError if such a decomposition is not
        possible.

    Examples
    --------

    Using Partition on 8 MPI ranks to split a periodic unit-cube

    >>> partition = Partition(1.0)
    >>> partition.rank
    0
    >>> partition.decomposition
    np.ndarray([2, 2, 2])
    >>> partition.coordinates
    np.ndarray([0, 0, 0])
    >>> partition.origin
    np.ndarray([0., 0., 0.])
    >>> partition.extent
    np.ndarray([0.5, 0.5, 0.5])


    """

    def __init__(
        self,
        dimensions: int = 3,
        *,
        comm: MPI.Intracomm | None = None,
        create_neighbor_topo: bool = False,
        commensurate_topo: list[int] | None = None,
    ) -> None:
        self._neighbor_topo: MPI.Distgraphcomm | None = None
        self._neighbor_ranks = None

        self._dimensions = dimensions
        self._mpi_init = False
        if comm is not None:
            self._comm = comm
        else:
            if not MPI.Is_initialized():
                MPI.Init()
                self._mpi_init = True
            self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._nranks = self._comm.Get_size()

        assert dimensions > 0
        assert isinstance(dimensions, int)

        if commensurate_topo is None:
            self._decomposition = MPI.Compute_dims(self._nranks, [0] * self._dimensions)
        else:
            nranks_factors = _factorize(self._nranks)
            decomposition, remainder = _distribute_factors(
                nranks_factors, commensurate_topo
            )
            assert np.all(decomposition * remainder == np.array(commensurate_topo))
            assert np.prod(decomposition) == self._nranks
            self._decomposition = decomposition.tolist()

        periodic = [True] * self._dimensions

        self._topo = self._comm.Create_cart(self._decomposition, periods=periodic)
        self._coords = list(self._topo.coords)

        self._neighbors = np.zeros([3] * self._dimensions, dtype=np.int32)
        for idx in itertools.product([-1, 0, 1], repeat=self._dimensions):
            coord = [
                (self._coords[d] + idx[d]) % self._decomposition[d]
                for d in range(self._dimensions)
            ]
            neigh = self._topo.Get_cart_rank(coord)
            self._neighbors[tuple(_i + 1 for _i in idx)] = neigh

        self._extent = [1.0 / self._decomposition[i] for i in range(self._dimensions)]
        self._origin = [
            self._coords[i] * self._extent[i] for i in range(self._dimensions)
        ]

        # A graph topology linking all neighbors
        if create_neighbor_topo:
            neighbors = np.unique(
                [n for n in self._neighbors.flatten() if n != self._rank]
            ).astype(np.int32)
            self._neighbor_topo = self._topo.Create_dist_graph_adjacent(
                sources=neighbors, destinations=neighbors, reorder=False
            )
            assert self._neighbor_topo.is_topo
            inout_neighbors = self._neighbor_topo.inoutedges
            assert len(inout_neighbors[0]) == len(inout_neighbors[1])
            for i in range(len(inout_neighbors[0])):
                if inout_neighbors[0][i] != inout_neighbors[1][i]:
                    print(
                        "neighbor topo: neighbors in sources and destinations are not ordered the same",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._topo.Abort()
            self._neighbor_ranks = inout_neighbors[0]

    def __del__(self) -> None:
        if self._neighbor_topo is not None:
            self._neighbor_topo.Free()
        if hasattr(self, "_topo"):
            self._topo.Free()
        if self._mpi_init:
            MPI.Finalize()

    @property
    def dimensions(self) -> int:
        """Dimension of the partitioned volume"""
        return self._dimensions

    @property
    def comm(self) -> MPI.Cartcomm:
        """3D Cartesian MPI Topology / Communicator"""
        return self._topo

    @property
    def comm_neighbor(self) -> MPI.Distgraphcomm | None:
        """Graph MPI Topology / Communicator, connecting the neighboring ranks
        (symmetric)"""
        return self._neighbor_topo

    @property
    def rank(self) -> int:
        """int: the MPI rank of this processor"""
        return self._rank

    @property
    def nranks(self) -> int:
        """int: the total number of processors"""
        return self._nranks

    @property
    def decomposition(self) -> list[int]:
        """np.ndarray: the decomposition of the cubic volume: number of ranks along each dimension"""
        return self._decomposition

    @property
    def coordinates(self) -> list[int]:
        """np.ndarray: 3D indices of this processor"""
        return self._coords

    @property
    def extent(self) -> list[float]:
        """np.ndarray: Length along each axis of this processors subvolume (same for all procs)"""
        return self._extent

    @property
    def origin(self) -> list[float]:
        """np.ndarray: Cartesian coordinates of the origin of this processor"""
        return self._origin

    def get_neighbor(self, di: list[int]) -> int:
        """get the rank of the neighbor at relative position (dx, dy, dz, ...)

        Parameters
        ----------

        di: List[int]
            list of relative coordinates, one of `[-1, 0, 1]`.
        """
        assert len(di) == self._dimensions
        return self._neighbors[np.array(di) + 1]

    @property
    def neighbors(self) -> np.ndarray:
        """np.ndarray: a 3^d dimensional array with the ranks of the neighboring processes
        (`neighbors[1,1,1, ...]` is this processor)"""
        return self._neighbors

    @property
    def neighbor_ranks(self) -> list[int] | None:
        """np.ndarray: a flattened list of the unique neighboring ranks"""
        return self._neighbor_ranks

    @property
    def ranklist(self) -> np.ndarray:
        """np.ndarray: A complete list of ranks, aranged by their coordinates.
        The array has shape `partition.decomposition`"""
        ranklist = np.empty(self.decomposition, dtype=np.int32)
        for idx in itertools.product(*map(range, self.decomposition)):
            ranklist[tuple(idx)] = self._topo.Get_cart_rank(idx)
        return ranklist
