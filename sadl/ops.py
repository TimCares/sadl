"""Tensor-level ops (not wrapped by numpy/cupy, they build their own graph nodes)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backend import (
    BACKEND,
    TensorDevice,
)
from .grad_mode import is_global_grad_mode_enabled
from .grad_ops import (
    astype_backward,
    copy_to_device_backward,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from .tensor import Tensor


def copy_to_device(x: Tensor, /, *, device: TensorDevice) -> Tensor:
    """Copy a Tensor to the specified device.

    This operation just creates a new tensor, so the actual copy
    operation is performed in the `sadl.tensor` factory function.
    However, this function adds:

        1. Device plausibility checks.
        2. Gradient tracking if `x` is in a computation graph.

    Args:
        x (Tensor): The Tensor to copy.
        device (TensorDevice): Target device, "cpu" or GPU id (int).

    Raises:
        ValueError: If using numpy backend and requesting a GPU device.

    Returns:
        Tensor: The Tensor on the target device, or the original if already there.
    """
    if x.device == device:
        return x
    if BACKEND == "numpy" and device.type == "cuda":
        raise ValueError(
            "Trying to copy to a gpu, but no gpu backend is available."
            " Please check cupy and gpu availability."
        )

    # Check if this is a non-leaf tensor in an active computation graph
    # (like intermediate activations in multi-GPU scenarios)
    is_in_graph = (
        is_global_grad_mode_enabled()
        and x.requires_grad
        and len(x.src) > 0
        and any(s.requires_grad for s in x.src)
    )

    from .tensor import tensor  # noqa: PLC0415

    new_tensor = tensor(
        x.data,
        device=device,
        dtype=x.dtype,
        requires_grad=True if is_in_graph else x.requires_grad,
        keep_grad=x.keep_grad,
    )

    if is_in_graph:
        # Tracked operation: gradients flow back through device transfer
        new_tensor.src = (x,)
        new_tensor.backward_fn = copy_to_device_backward

    return new_tensor


# TODO
def astype(x: Tensor, /, *, dtype: npt.DTypeLike) -> Tensor:
    """Create a copy of a Tensor `x` with its dtype given by `dtype`.

    Args:
        x (Tensor): The Tensor to have as `dtype`.
        dtype (npt.DTypeLike): The dtype.

    Returns:
        Tensor: A copy of `x` with the dtype given by `dtype`.
    """
    if x.dtype == dtype:
        return x

    # Check if this is a non-leaf tensor in an active computation graph
    # (like intermediate activations in multi-GPU scenarios)
    is_in_graph = (
        is_global_grad_mode_enabled()
        and x.requires_grad
        and len(x.src) > 0
        and any(s.requires_grad for s in x.src)
    )

    from .tensor import tensor  # noqa: PLC0415

    new_tensor = tensor(
        x.data,
        device=x.device,
        dtype=dtype,
        requires_grad=True if is_in_graph else x.requires_grad,
        keep_grad=x.keep_grad,
    )

    if is_in_graph:
        # Tracked operation: gradients flow back through dtype conversion
        new_tensor.src = (x,)
        new_tensor.backward_fn = astype_backward

    return new_tensor


__all__ = [
    "astype",
    "copy_to_device",
]
