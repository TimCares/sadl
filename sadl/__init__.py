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
    SupportsCupyDevice,
    TensorDevice,
    get_rng,
)
from .backend.dtype import (
    Bool,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    TensorDType,
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
from .ops import (
    absolute,
    add,
    argmax,
    argmin,
    cos,
    divide,
    exp,
    log,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    multiply,
    negative,
    power,
    reshape,
    sin,
    sqrt,
    square,
    subtract,
    sum,
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
    "Bool",
    "DeviceLike",
    "DeviceType",
    "Float16",
    "Float32",
    "Float64",
    "Function",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
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
    "TensorDType",
    "TensorDevice",
    "__version__",
    "absolute",
    "add",
    "argmax",
    "argmin",
    "cos",
    "divide",
    "exp",
    "eye",
    "get_rng",
    "grad_ops",
    "is_global_grad_mode_enabled",
    "load",
    "log",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "multiply",
    "negative",
    "no_grad",
    "no_grad_fn",
    "ones",
    "ones_like",
    "power",
    "reshape",
    "save",
    "set_global_grad_mode",
    "sin",
    "sqrt",
    "square",
    "subtract",
    "sum",
    "tensor",
    "zeros",
    "zeros_like",
]
