from .backend import BACKEND, TensorDevice, xp


def copy_array(array: "xp.ndarray", device: TensorDevice) -> "xp.ndarray":
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
    if isinstance(device, int):
        with xp.cuda.Device(device):
            return xp.asarray(array)
    else:
        return xp.asnumpy(array)
