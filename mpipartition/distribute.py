import sys
from typing import List, Union

import numpy as np

from .partition import Partition
from ._send_home import distribute_dataset_by_home

ParticleDataT = dict[str, np.ndarray]


def distribute(
    partition: Partition,
    box_size: float,
    data: ParticleDataT,
    coord_keys: List[str],
    *,
    verbose: Union[bool, int] = False,
    verify_count: bool = True,
    all2all_iterations: int = 1,
) -> ParticleDataT:
    """Distribute data among MPI ranks according to data position and volume partition

    The position of each TreeData element is given by the data columns
    specified with `coord_keys`.

    Parameters
    ----------

    partition:
        The MPI partition defining which rank should own which subvolume of the
        data

    box_size:
        The size of the full simulation volume

    data:
        The treenode / coretree data that should be distributed

    coord_keys:
        The columns in `data` that define the position of the object

    verbose:
        If True, print summary statistics of the distribute. If > 1, print
        statistics of each rank (i.e. how much data each rank sends to every
        other rank).

    verify_count:
        If True, make sure that total number of objects is conserved

    all2all_iterations:
        The number of iterations to use for the all-to-all communication.
        This is useful for large datasets, where MPI_Alltoallv may fail

    Returns
    -------
    data: ParticleDataT
        The distributed particle data (i.e. the data that this rank owns)

    """
    assert len(coord_keys) == partition.dimensions

    # get some MPI and partition parameters
    nranks = partition.nranks
    if nranks == 1:
        return data

    # rank = partition.rank
    comm = partition.comm
    dimensions = partition.dimensions
    ranklist = np.array(partition.ranklist)
    extent = box_size * np.array(partition.extent)

    # count number of particles we have
    total_to_send = len(data[coord_keys[0]])

    if total_to_send > 0:
        # Check validity of coordinates
        for i in range(dimensions):
            _x = data[coord_keys[i]]
            _min = _x.min()
            _max = _x.max()
            if _min < 0 or _max > box_size:
                print(
                    f"Error in distribute: position {coord_keys[i]} out of range: [{_min}, {_max}]",
                    file=sys.stderr,
                    flush=True,
                )
                comm.Abort()

        # Find home of each particle
        idx = np.array(
            [data[coord_keys[i]] / extent[i] for i in range(dimensions)]
        ).astype(np.int32)
        idx = np.clip(idx, 0, np.array(partition.decomposition)[:, np.newaxis] - 1)
        home_idx = ranklist[tuple(idx)]
    else:
        # there are no particles on this rank
        home_idx = np.empty(0, dtype=np.int32)

    data_new = distribute_dataset_by_home(
        partition,
        data,
        home_idx=home_idx,
        verbose=verbose,
        verify_count=verify_count,
        all2all_iterations=all2all_iterations,
    )

    return data_new
