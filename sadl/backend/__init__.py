"""Backend for all ops in SADL."""

from .array_provider import (
    BACKEND,
    ArrayModule,
    NDArray,
    NDArrayLike,
    get_array_module_from_device,
    is_ndarray,
)
from .device import (
    DeviceLike,
    DeviceType,
    SupportsCupyDevice,
    TensorDevice,
)
from .utils import (
    copy_array,
)

__all__ = [
    "BACKEND",
    "ArrayModule",
    "DeviceLike",
    "DeviceType",
    "NDArray",
    "NDArrayLike",
    "SupportsCupyDevice",
    "TensorDevice",
    "copy_array",
    "get_array_module_from_device",
    "is_ndarray",
]
