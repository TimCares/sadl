"""Backend for all ops in SADL."""

from .array_provider import (
    BACKEND,
    ArrayModule,
    NDArray,
    NDArrayLike,
)
from .device import (
    DeviceLike,
    DeviceType,
    SupportsCupyDevice,
    TensorDevice,
)
from .dtype import (
    TensorDType,
)
from .random import (
    get_rng,
)
from .utils import (
    copy_array,
    get_array_module,
    get_array_module_from_device,
)

__all__ = [
    "BACKEND",
    "ArrayModule",
    "DeviceLike",
    "DeviceType",
    "NDArray",
    "NDArrayLike",
    "SupportsCupyDevice",
    "TensorDType",
    "TensorDevice",
    "copy_array",
    "get_array_module",
    "get_array_module_from_device",
    "get_rng",
]
