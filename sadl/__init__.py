"""SADL: Simple Autograd Deep Learning.

A minimal, readable deep learning framework built on NumPy/CuPy.
"""

from importlib.metadata import PackageNotFoundError, version

from . import grad_ops

try:
    __version__ = version("py-sadl")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for uninstalled package
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
    LogSoftmax,
    Mlp,
    ReLU,
    Sigmoid,
    Softmax,
)
from .ops import (
    copy_to_device,
    ones_like,
    zeros_like,
)
from .optimizer import (
    SGD,
    Adam,
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
    "Adam",
    "Function",
    "Linear",
    "LogSoftmax",
    "Mlp",
    "Optimizer",
    "Parameter",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tensor",
    "TensorDevice",
    "__version__",
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
