"""SADL: Simple Autograd Deep Learning.

A minimal, readable deep learning framework built on NumPy/CuPy.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("py-sadl")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for uninstalled package
from .backend import (
    BACKEND,
    DeviceLike,
    DeviceType,
    SupportsCupyDevice,
    TensorDevice,
    copy_array,
    grad_ops,
    is_global_grad_mode_enabled,
    no_grad,
    no_grad_fn,
    normalize_device,
    set_global_grad_mode,
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
from .optimizer import (
    SGD,
    Adam,
    Optimizer,
)
from .tensor import (
    Parameter,
    Tensor,
    tensor,
)
from .utils import (
    ones_like,
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
    "Optimizer",
    "Parameter",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "SupportsCupyDevice",
    "Tensor",
    "TensorDevice",
    "__version__",
    "copy_array",
    "grad_ops",
    "is_global_grad_mode_enabled",
    "load",
    "no_grad",
    "no_grad_fn",
    "normalize_device",
    "ones_like",
    "save",
    "set_global_grad_mode",
    "tensor",
    "xp",
    "zeros_like",
]
