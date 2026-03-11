"""SADL: Simple Autograd Deep Learning.

A minimal, readable deep learning framework built on NumPy/CuPy.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("py-sadl")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for uninstalled package
from . import grad_ops
from .backend import (
    BACKEND,
    DeviceLike,
    DeviceType,
    NDArray,
    NDArrayLike,
    SupportsCupyDevice,
    TensorDevice,
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
from .grad_mode import (
    is_global_grad_mode_enabled,
    no_grad,
    no_grad_fn,
    set_global_grad_mode,
)
from .optimizer import (
    SGD,
    Adam,
    Optimizer,
)
from .tensor import (
    Parameter,
    Tensor,
    eye,
    ones,
    ones_like,
    tensor,
    zeros,
    zeros_like,
)

__all__ = [
    "BACKEND",
    "SGD",
    "Adam",
    "DeviceLike",
    "DeviceType",
    "Function",
    "Linear",
    "LogSoftmax",
    "Mlp",
    "NDArray",
    "NDArrayLike",
    "Optimizer",
    "Parameter",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "SupportsCupyDevice",
    "Tensor",
    "TensorDevice",
    "__version__",
    "eye",
    "grad_ops",
    "is_global_grad_mode_enabled",
    "load",
    "no_grad",
    "no_grad_fn",
    "ones",
    "ones_like",
    "save",
    "set_global_grad_mode",
    "tensor",
    "zeros",
    "zeros_like",
]
