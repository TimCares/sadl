"""Utility backend functions."""

import numpy as np
import numpy.typing as npt

from .array_provider import BACKEND, NDArray, NDArrayLike, xp
from .device import TensorDevice


def copy_array(
    array: NDArrayLike,
    device: TensorDevice | None = None,
    dtype: npt.DTypeLike | None = None,
) -> NDArray:
    """Copy an ndarray (or array-like data).

    Can be used to copy an array to another device if `device` is not the same
    as the device of `array`. If it is the same, an ordinary copy is created.

    Args:
        array (NDArrayLike): The array to copy.
        device (TensorDevice | None, optional): Target device. If None, device will
            be the device on which `array` is located. Defaults to None.
        dtype (npt.DTypeLike | None, optional): The target dtype. If None, will be inferred
            from `array`, so the dtype stays the same. Defaults to None.

    Raises:
        ValueError: If using the numpy backend and a CUDA device is requested.

    Returns:
        NDArray: The copy of the array.
    """
    if device is None:
        device = TensorDevice.create(getattr(array, "device", "cpu"))

    if BACKEND == "numpy" and device.type == "cuda":
        raise ValueError(
            "Trying to copy to a gpu, but no gpu backend is available. "
            "Please check cupy and gpu availability."
        )

    if device.type == "cuda":
        with xp.cuda.Device(device.device_id):  # type: ignore[attr-defined]
            return xp.array(array, dtype=dtype)  # type: ignore[attr-defined]
    cpu_array = xp.asnumpy(array) if BACKEND == "cupy" else array  # type: ignore[attr-defined] # first move to cpu
    return np.array(cpu_array, dtype=dtype)


__all__ = [
    "copy_array",
]
