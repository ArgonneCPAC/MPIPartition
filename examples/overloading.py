from mpipartition import Partition, overload
import numpy as np

def test_overload(dimensions):
    assert dimensions < 7
    assert dimensions > 0
    labels = "xyzuvw"[:dimensions]

    partition = Partition(dimensions)
    rank = partition.rank
    nranks = partition.nranks

    np.random.seed(rank)
    N = 2
    data = {
        x: np.random.uniform(0, 1, N)*partition.extent[i] + partition.origin[i] for i, x in enumerate(labels)
    }
    data['s'] =  rank * np.ones(N, dtype=np.uint16)

    for i in range(nranks):
        if partition.rank == i:
            print(f"Rank {i}")
            print(" - extent: ", partition.extent)
            print(" - origin: ", partition.origin)
            # print(" - neighbors:\n", partition.neighbors)
            for i in range(N):
                pos_str = ', '.join(f"{data[x][i]:.2f}" for x in labels)
                print(f" - {i}: [{pos_str}]")
            print("", flush=True)

    data = overload(partition, 1., data, 0.2, labels, verbose=2)

    N = len(data['x'])
    for i in range(nranks):
        if partition.rank == i:
            print(f"Rank {i}")
            print(" - extent: ", partition.extent)
            print(" - origin: ", partition.origin)
            for i in range(N):
                pos_str = ', '.join(f"{data[x][i]:.2f}" for x in labels)
                s = data['s'][i]
                inside =  s == rank
                print(f" - {i}: [{pos_str} | {s:3d}] | {'o' if not inside else ''}")
            print("", flush=True)

if __name__ == "__main__":
    # test_distribute()
    test_overload(2)