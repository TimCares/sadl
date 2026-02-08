from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

logger = logging.getLogger(__name__)


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
except (ImportError, RuntimeError) as err:
    import numpy as xp

    BACKEND = "numpy"
    logger.warning("Cupy backend unavailable; falling back to numpy (cpu)")
    logger.debug(f"Falling back to numpy because: {err!r}")


DeviceType = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class TensorDevice:
    """The global device identifier for sadl Tensors."""

    type: DeviceType
    device_id: int = 0


@runtime_checkable
class SupportsCupyDevice(Protocol):
    """Cupy protocol to access cuda device `id`."""

    id: int  # cupy.cuda.Device exposes attribute "id"


DeviceLike: TypeAlias = TensorDevice | Literal["cpu"] | int | SupportsCupyDevice


def normalize_device(device: DeviceLike) -> TensorDevice:
    """Transforms any device-like type into a sadl TensorDevice type.

    Args:
        device (DeviceLike): The device candidate.

    Raises:
        TypeError: If `device` denotes an unsupported device.
            Should not happen.

    Returns:
        TensorDevice: The sadl TensorDevice.
    """
    if isinstance(device, TensorDevice):
        return device
    if device == "cpu":
        return TensorDevice("cpu")
    if isinstance(device, int):
        return TensorDevice("cuda", device)

    if hasattr(device, "id"):
        return TensorDevice("cuda", int(device.id))

    raise TypeError(f"Unsupported device spec: {device!r}")


__all__ = [
    "BACKEND",
    "DeviceLike",
    "DeviceType",
    "SupportsCupyDevice",
    "TensorDevice",
    "normalize_device",
    "xp",
]
