"""All custom or extended (from numpy/cupy) operations on SADL Tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backend import BACKEND, TensorDevice, xp

if TYPE_CHECKING:
    pass


def copy_array(array: xp.ndarray, device: TensorDevice) -> xp.ndarray:
    """Copy an array to the specified device.

    Args:
        array (xp.ndarray): The array to copy.
        device (TensorDevice): Target device, "cpu" or GPU id (int).

    Raises:
        ValueError: If device string is not "cpu".
        ValueError: If using numpy backend and requesting a GPU device.

    Returns:
        xp.ndarray: The array on the target device, or the original if already there.
    """
    if array.device == device:
        return array
    if isinstance(device, str) and device != "cpu":
        raise ValueError('Only "cpu" allowed as string device.')
    if BACKEND == "numpy":
        raise ValueError(
            "Copying to another device is only possible when using cupy "
            "as the backend. Currently, numpy is the backend. Please "
            "check cupy and gpu availability."
        )
    # cupy:
    if device.type == "cuda":
        with xp.cuda.Device(device.device_id):
            return xp.asarray(array)
    else:
        return xp.asnumpy(array)


__all__ = [
    "copy_array",
]
