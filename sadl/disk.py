"""Code for serializing and deserializing data."""

from __future__ import annotations

import struct
from collections import OrderedDict
from typing import Any

import numpy as np

from .tensor import Tensor, tensor

_SADL_MAGIC = b"SADL"
_SADL_VERSION = 1


def _dtype_to_str(dtype: Any) -> str:
    """Convert numpy/cupy dtype to string representation."""
    return str(np.dtype(dtype).name)


def _str_to_dtype(dtype_str: str) -> Any:
    """Convert string back to numpy dtype."""
    return np.dtype(dtype_str)


def save(data: Tensor | OrderedDict[str, Tensor], file_path: str) -> None:
    """Save Tensor data to disk using custom binary format.

    Args:
        data (Tensor | OrderedDict[str, Tensor]): The data to save,
            can either be a single Tensor or an OrderedDict with strings
            as keys and Tensors as values.
        file_path (str): The file path to which to store the data. Must
            end with ".sadl".

    Raises:
        ValueError: If an OrderedDict with non-Tensor values is passed.
        ValueError: If file_path doesn't end with ".sadl".
    """
    if not file_path.endswith(".sadl"):
        raise ValueError('file_path must end with ".sadl"')

    # Normalize to OrderedDict
    if isinstance(data, Tensor):
        tensors = OrderedDict([("__single__", data)])
    else:
        if not all(isinstance(v, Tensor) for v in data.values()):
            raise ValueError("If an OrderedDict is passed, all values must be Tensors.")
        tensors = data

    with open(file_path, "wb") as f:
        # Write header
        f.write(_SADL_MAGIC)
        f.write(struct.pack("<B", _SADL_VERSION))  # uint8 version
        f.write(struct.pack("<I", len(tensors)))  # uint32 num tensors

        # Write each tensor
        for key, tensor in tensors.items():
            # Convert to numpy (CPU) for serialization
            arr = np.asarray(tensor)
            # Ensure C-contiguous
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)

            # Key
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<I", len(key_bytes)))  # uint32 key length
            f.write(key_bytes)

            # Dtype
            dtype_str = _dtype_to_str(arr.dtype)
            dtype_bytes = dtype_str.encode("utf-8")
            f.write(struct.pack("<B", len(dtype_bytes)))  # uint8 dtype length
            f.write(dtype_bytes)

            # Shape
            f.write(struct.pack("<B", arr.ndim))  # uint8 ndim
            f.writelines(struct.pack("<Q", dim) for dim in arr.shape)  # uint64 per dimension

            # Raw data
            f.write(arr.tobytes())


def load(file_path: str) -> Tensor | OrderedDict[str, Tensor]:
    """Load Tensor data from disk.

    Args:
        file_path (str): The file path from which to read the data. Must
            end with ".sadl".

    Raises:
        ValueError: If file_path doesn't end with ".sadl".
        ValueError: If file has invalid magic bytes or unsupported version.

    Returns:
        Tensor | OrderedDict[str, Tensor]: The loaded data. Returns a single
            Tensor if one was saved, otherwise an OrderedDict.
    """
    if not file_path.endswith(".sadl"):
        raise ValueError('file_path must end with ".sadl"')

    with open(file_path, "rb") as f:
        # Read and validate header
        magic = f.read(4)
        if magic != _SADL_MAGIC:
            raise ValueError(f"Invalid file format. Expected SADL magic bytes, got {magic!r}")

        version = struct.unpack("<B", f.read(1))[0]
        if version != _SADL_VERSION:
            raise ValueError(f"Unsupported version {version}. Expected {_SADL_VERSION}")

        num_tensors = struct.unpack("<I", f.read(4))[0]

        # Read tensors
        tensors: OrderedDict[str, Tensor] = OrderedDict()
        for _ in range(num_tensors):
            # Key
            key_length = struct.unpack("<I", f.read(4))[0]
            key = f.read(key_length).decode("utf-8")

            # Dtype
            dtype_length = struct.unpack("<B", f.read(1))[0]
            dtype_str = f.read(dtype_length).decode("utf-8")
            dtype = _str_to_dtype(dtype_str)

            # Shape
            ndim = struct.unpack("<B", f.read(1))[0]
            shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(ndim))

            # Data
            num_bytes = int(np.prod(shape)) * dtype.itemsize
            data_bytes = f.read(num_bytes)
            arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

            tensors[key] = tensor(arr)

        # Return single tensor if that's what was saved
        if len(tensors) == 1 and "__single__" in tensors:
            return tensors["__single__"]
        return tensors


__all__ = [
    "load",
    "save",
]
