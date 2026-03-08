"""Device abstraction for sadl Tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

DeviceType = Literal["cpu", "cuda"]


@runtime_checkable
class SupportsCupyDevice(Protocol):
    """Cupy protocol to access cuda device `id`."""

    id: int  # cupy.cuda.Device exposes attribute "id"


@dataclass(frozen=True)
class TensorDevice:
    """The global device identifier for sadl Tensors."""

    type: DeviceType
    device_id: int = 0

    @classmethod
    def create(cls, device: DeviceLike) -> TensorDevice:
        """Transforms any device-like value into a sadl `TensorDevice`.

        Args:
            device (DeviceLike): A `TensorDevice`, the string `"cpu"`, a
                CUDA device index (`int`), or any object with an `id`
                attribute (e.g. `cupy.cuda.Device`).

        Raises:
            TypeError: If `device` is not a recognised device specification.

        Returns:
            TensorDevice: The normalised sadl device.
        """
        if isinstance(device, cls):
            return device
        if device == "cpu":
            return cls("cpu")
        if isinstance(device, int):
            return cls("cuda", device)
        if hasattr(device, "id"):
            return cls("cuda", int(getattr(device, "id", 0)))

        raise TypeError(f"Unsupported device spec: {device!r}")


DeviceLike: TypeAlias = TensorDevice | Literal["cpu"] | int | SupportsCupyDevice


__all__ = [
    "DeviceLike",
    "DeviceType",
    "SupportsCupyDevice",
    "TensorDevice",
]
