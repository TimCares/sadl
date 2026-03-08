"""NDArray provider abstraction for sadl data buffer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np

from .device import DeviceLike, DeviceType, SupportsCupyDevice, TensorDevice

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


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


try:
    import cupy as xp  # type: ignore[import-untyped]

    _validate_cupy_available()

    BACKEND = "cupy"
    logger.debug("Using cupy as backend")
except (ImportError, RuntimeError) as err:
    import numpy as xp

    BACKEND = "numpy"
    logger.warning("Cupy backend unavailable; falling back to numpy (cpu)")
    logger.debug(f"Falling back to numpy because: {err!r}")


NDArray: TypeAlias = np.ndarray
NDArrayLike: TypeAlias = NDArray | int | float | list[Any]


class ArrayModule(Protocol):
    """Structural type for numpy/cupy module objects used as array backends.

    Attribute access is typed as a generic array operation so that
    `__backend__.some_func(x)` returns `NDArray` without per-function
    stubs for every numpy/cupy function we use.
    """

    def __getattr__(self, __name: str) -> Callable[..., NDArray]: ...


__all__ = [
    "BACKEND",
    "ArrayModule",
    "DeviceLike",
    "DeviceType",
    "NDArray",
    "NDArrayLike",
    "SupportsCupyDevice",
    "TensorDevice",
]
