"""NDArray provider abstraction for sadl data buffer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeGuard, cast

import numpy as np
from numpy.typing import ArrayLike as NDArrayLike
from numpy.typing import NDArray as _NDArrayBase

from .device import DeviceLike, DeviceType, SupportsCupyDevice, TensorDevice

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable


NDArray: TypeAlias = _NDArrayBase[Any]


def _validate_cupy_available() -> None:
    """Validate that CuPy is available with working CUDA devices.

    Raises:
        RuntimeError: If CUDA is unavailable or no devices are found.
    """
    try:
        _device_count = xp.cuda.runtime.getDeviceCount()  # type: ignore[attr-defined]
    except Exception as exc:
        raise RuntimeError("Cupy is installed but CUDA is unavailable") from exc

    if _device_count < 1:
        raise RuntimeError("Cupy is installed but no CUDA devices are available")


_cupy_ndarray_type: type[Any] | None = None

try:
    import cupy as xp  # type: ignore[import-untyped]

    _validate_cupy_available()
    _cupy_ndarray_type = cast("type[Any]", xp.ndarray)  # type: ignore[reportUnknownMemberType]

    BACKEND = "cupy"
    logger.debug("Using cupy as backend")
except (ImportError, RuntimeError) as err:
    import numpy as xp

    BACKEND = "numpy"
    logger.warning("Cupy backend unavailable; falling back to numpy (cpu)")
    logger.debug(f"Falling back to numpy because: {err!r}")


_RUNTIME_ARRAY_TYPES: tuple[type[Any], ...] = (
    (np.ndarray,) if _cupy_ndarray_type is None else (np.ndarray, _cupy_ndarray_type)
)


def get_array_module_from_device(device: TensorDevice) -> ArrayModule:
    """Get the array module (backend) based on a device.

    Args:
        device (TensorDevice): The device.

    Returns:
        ArrayModule: The array module used for arrays on `device`.
    """
    if device.type == "cpu":
        return cast("ArrayModule", np)
    if BACKEND == "numpy":
        raise RuntimeError("Cannot run on cuda without a cupy backend.")
    return cast("ArrayModule", xp)  # xp will by cupy at this point


def is_ndarray(data: Any) -> TypeGuard[NDArray]:
    """Return whether `value` is a supported concrete backend array instance."""
    return isinstance(data, _RUNTIME_ARRAY_TYPES)


def is_ndarray_like(data: Any) -> TypeGuard[NDArrayLike]:
    """Return whether `data` is a numeric scalar or array that can serve as an operand."""
    return isinstance(data, (int, float, complex, np.generic)) or is_ndarray(data)


class ArrayModule(Protocol):
    """Structural type for numpy/cupy module objects used as array backends.

    Attribute access is typed as a generic array operation so that
    `array_module.some_func(x)` returns `NDArray` without per-function
    stubs for every numpy/cupy function we use.
    """

    def __getattr__(self, name: str) -> Callable[..., NDArray]:
        """Get the function with name `name`.

        Args:
            name (str): The name of the function.

        Returns:
            Callable[..., NDArray]: The array operation returning an array.

        Examples:
            >>> array_module.max(x)

            >>> np = array_module
            ... np.argmin(x)
        """
        ...


__all__ = [
    "BACKEND",
    "ArrayModule",
    "DeviceLike",
    "DeviceType",
    "NDArray",
    "NDArrayLike",
    "SupportsCupyDevice",
    "TensorDevice",
    "get_array_module_from_device",
    "is_ndarray",
    "is_ndarray_like",
]
