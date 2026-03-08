"""Provider abstraction for random number generation.

Which provider is needed can only be determined at runtime,
as it depends on the devices (gpu) that are available.

If client code were to use `xp`, the behavior could change
based on the runtime: If a gpu is available and cupy is installed,
xp would create Tensors on the gpu, if only numpy (or no gpu) is
available, random Tensors would be created on the cpu.

Creating a getter for the random number generator that requries
a device as an argument enforces "device-awareness" at the client,
reducing client surprise.
If a device is specified that does not exist, an error is raised before
getting the generator.
"""

import numpy as np

from .array_provider import BACKEND, xp
from .device import TensorDevice


def get_rng(device: TensorDevice | None = None) -> np.random.Generator:
    """Retrieve the `default_rng()` for `device`.

    Will be from numpy if the device is cpu, else cupy.

    Note: One could always get the cpu/numpy rng and then copy the created
    Tensors to gpu, but this adds a roundtrip.

    Args:
        device (TensorDevice | None, optional): The device for which to get the generator.
            If None, cpu will be used. Defaults to None.

    Raises:
        RuntimeError: If a gpu device is provided but no gpu/cupy is available.

    Returns:
        np.random.Generator: The random number generator. Due to duck-typing we
            can use np even if cupy is used.
    """
    if device is None:
        device = TensorDevice("cpu")

    if BACKEND == "numpy" and device.type == "cuda":
        raise RuntimeError(
            "Trying to use a random number generator for device gpu, "
            "but no gpu backend is available. "
            "Please check cupy and gpu availability."
        )
    if device.type == "cuda":
        with xp.cuda.Device(device.device_id):  # type: ignore[attr-defined]
            return xp.random.default_rng()  # type: ignore[attr-defined]
    return np.random.default_rng()


__all__ = [
    "get_rng",
]
