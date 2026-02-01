import logging
from typing import TYPE_CHECKING, Any, Literal

logger = logging.getLogger(__name__)


TensorDevice = Literal["cpu"] | int


if TYPE_CHECKING:
    import numpy

    ModuleType = numpy.ndarray[Any, Any] | Any  # numpy or cupy module


def _validate_cupy_available() -> None:
    """Validate that CuPy is available with working CUDA devices.

    Raises:
        RuntimeError: If CUDA is unavailable or no devices are found.
    """
    try:
        _device_count = xp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise RuntimeError("Cupy is installed but CUDA is unavailable") from exc

    if _device_count < 1:
        raise RuntimeError("Cupy is installed but no CUDA devices are available")


try:
    import cupy as xp

    _validate_cupy_available()

    BACKEND = "cupy"
    logger.debug("Using cupy as backend")
except (ImportError, RuntimeError):
    import numpy as xp

    BACKEND = "numpy"
    logger.warning("Cupy backend unavailable; falling back to numpy (cpu)")


__all__ = ["BACKEND", "TensorDevice", "xp"]
