"""SADL: Simple Autograd Deep Learning.

A minimal, readable deep learning framework built on NumPy/CuPy.
"""

from . import grad_ops
from .backend import (
    BACKEND,
    TensorDevice,
    xp,
)
from .disk import (
    load,
    save,
)
from .function import (
    Function,
    Linear,
    Mlp,
    ReLU,
    Sigmoid,
)
from .ops import (
    copy_to_device,
    ones_like,
    zeros_like,
)
from .optimizer import (
    SGD,
    Optimizer,
)
from .tensor import (
    Parameter,
    Tensor,
    get_current_global_grad_mode,
    no_grad,
    no_grad_fn,
    set_global_grad_mode,
    tensor,
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
    "copy_to_device",
    "get_current_global_grad_mode",
    "grad_ops",
    "load",
    "no_grad",
    "no_grad_fn",
    "ones_like",
    "save",
    "set_global_grad_mode",
    "tensor",
    "xp",
    "zeros_like",
]
