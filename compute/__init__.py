import numpy as np
try:
    import torch

    def add_torch(a: torch.Tensor, b: torch.Tensor):
        return a + b
except:
    print("[Warning] You're using the numpy version")


def add(a: np.ndarray, b: np.ndarray):
    return a + b


def add_float():
    print("add_float")
