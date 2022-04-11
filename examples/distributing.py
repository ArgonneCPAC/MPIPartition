from mpipartition import Partition, distribute
import numpy as np

def test_distribute(dimensions, N):
    partition = Partition(dimensions)
    assert dimensions < 7
    assert dimensions > 0
    labels = "xyzuvw"[:dimensions]

    data = {x: np.random.uniform(0, 1, N) for x in labels}
    data = distribute(partition, 1., data, labels)

    valid = np.ones(len(data[labels[0]]), dtype=np.bool_)
    for i, label in enumerate(labels):
        valid &= data[label] >= partition.origin[i]
        valid &= data[label] < partition.origin[i] + partition.extent[i]
    local_all_valid = np.all(valid)
    global_all_valid = np.all(partition.comm.allgather(local_all_valid))
    if partition.rank == 0:
        print(f"Passed for dim={dimensions} and {N} particles per rank: {global_all_valid}")

if __name__ == "__main__":
    test_distribute(1, 1000)
    test_distribute(2, 100)
    test_distribute(3, 10)
    test_distribute(4, 5)