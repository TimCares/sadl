"""Utility backend functions."""

from typing import cast

import numpy as np

from .array_provider import BACKEND, ArrayModule, NDArray, NDArrayLike, xp
from .device import TensorDevice
from .dtype import TensorDType


def copy_array(
    array: NDArrayLike,
    device: TensorDevice | None = None,
    dtype: TensorDType | None = None,
) -> NDArray:
    """Copy an ndarray (or array-like data).

    Can be used to copy an array to another device if `device` is not the same
    as the device of `array`. If it is the same, an ordinary copy is created.

    Args:
        array (NDArray): The array to copy.
        device (TensorDevice | None, optional): Target device. If None, device will
            be the device on which `array` is located. Defaults to None.
        dtype (TensorDType | None, optional): The target dtype. If None, will be inferred
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

    backend_dtype = (
        TensorDType.from_ndarraylike(array).to_backend() if dtype is None else dtype.to_backend()
    )

    if device.type == "cuda":
        with xp.cuda.Device(device.device_id):  # type: ignore[attr-defined]
            return xp.array(array, dtype=backend_dtype)  # type: ignore[attr-defined]
    cpu_array = xp.asnumpy(array) if BACKEND == "cupy" else array  # type: ignore[attr-defined] # first move to cpu
    return np.array(cpu_array, dtype=backend_dtype)


def get_array_module(array: NDArray) -> ArrayModule:
    """Return the array library (numpy or cupy) that owns `array`.

    When the numpy backend is active, every array is a `numpy.ndarray`
    and this always returns `numpy`. When the cupy backend is active,
    tensors can live on either device: GPU tensors are `cupy.ndarray`
    objects while CPU tensors that were moved with `copy_array` are
    plain `numpy.ndarray` objects. In that mixed case we delegate to
    `cupy.get_array_module`, which returns `cupy` for cupy arrays
    and `numpy` for everything else.

    Using the result as the `xp` module inside gradient operations
    guarantees that every array operation (`xp.sum`, `xp.exp`, ...)
    is dispatched to the library that owns the memory, avoiding illegal
    cross-library calls such as passing a numpy array into a cupy kernel.

    Args:
        array (NDArray): The array whose owning library should be returned.

    Returns:
        ArrayModule: The `numpy` or `cupy` module appropriate for `array`.
    """
    if BACKEND == "numpy":
        return cast("ArrayModule", np)
    return cast("ArrayModule", xp.get_array_module(array))  # type: ignore[attr-defined]


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


__all__ = [
    "get_array_module",
    "get_array_module_from_device",
]
