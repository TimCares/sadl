"""Backend for all ops in SADL."""

from .backend import (
    BACKEND,
    DeviceLike,
    DeviceType,
    SupportsCupyDevice,
    TensorDevice,
    normalize_device,
    xp,
)
from .grad_mode import (
    is_global_grad_mode_enabled,
    no_grad,
    no_grad_fn,
    set_global_grad_mode,
)
from .grad_ops import (
    GradOp,
    get_grad_op,
    normalize_grad_op_name,
)
from .ops import (
    copy_array,
)

__all__ = [
    "BACKEND",
    "DeviceLike",
    "DeviceType",
    "GradOp",
    "SupportsCupyDevice",
    "TensorDevice",
    "copy_array",
    "get_grad_op",
    "is_global_grad_mode_enabled",
    "no_grad",
    "no_grad_fn",
    "normalize_device",
    "normalize_grad_op_name",
    "set_global_grad_mode",
    "xp",
]
