from typing import Mapping, List, Union
import numpy as np

from .s2_partition import S2Partition

ParticleDataT = Mapping[str, np.ndarray]


def s2_distribute(
    partition: S2Partition,
    data: ParticleDataT,
    coord_keys: List[str],
    *,
    verbose: Union[bool, int] = False,
    verify_count: bool = True,
) -> ParticleDataT:
    pass
