"""SADL: Simple Autograd Deep Learning.

A minimal, readable deep learning framework built on NumPy/CuPy.
"""

from .src import (
    BACKEND,
    SGD,
    Function,
    Linear,
    Mlp,
    Optimizer,
    Parameter,
    ReLU,
    Sigmoid,
    Tensor,
    TensorDevice,
    grad_ops,
    load,
    no_grad,
    no_grad_fn,
    ones_like,
    save,
    tensor,
    zeros_like,
)

__all__ = [
    "BACKEND",
    "SGD",
    "Function",
    "Linear",
    "Mlp",
    "Optimizer",
    "Parameter",
    "ReLU",
    "Sigmoid",
    "Tensor",
    "TensorDevice",
    "grad_ops",
    "load",
    "no_grad",
    "no_grad_fn",
    "ones_like",
    "save",
    "tensor",
    "zeros_like",
]
