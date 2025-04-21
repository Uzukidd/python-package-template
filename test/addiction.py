import numpy as np
import torch
import compute
import compute.op
import compute.op.vector_compute

def main():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 8.0])
    print(compute.add(a, b))

    a = torch.from_numpy(a).float().cuda()
    b = torch.from_numpy(b).float().cuda()
    print(compute.op.vector_compute.add_vectors(a, b))

if __name__ == "__main__":
    main()