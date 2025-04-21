import torch

from . import vector_add


def add_vectors(a, b):
    return vector_add.vector_add_cuda(a, b)


__all__ = ['add_vectors']
