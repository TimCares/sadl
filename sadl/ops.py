"""All custom or extended (from numpy/cupy) operations on SADL Tensors."""

from __future__ import annotations

from typing import Any

from .backend import xp
from .tensor import (
    _GRAD_MODE_ENABLED,
    Tensor,
)
from .tensor import (
    _copy_to_device as copy_to_device,
)


def ones_like(
    other: Tensor,
    *,
    dtype: Any = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor of ones with the same shape and device as `other`.

    Args:
        other (Tensor): The tensor to match shape and device from.
        dtype (Any): Override dtype. Defaults to None (use other's dtype).
        requires_grad (bool): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: A tensor of ones.
    """
    # Use xp.ones(shape) instead of xp.ones_like(tensor) to avoid
    # triggering __array_function__ on the Tensor
    result: Tensor = xp.ones(other.shape, dtype=dtype or other.dtype).view(Tensor)
    result.requires_grad = _GRAD_MODE_ENABLED and requires_grad
    return result


def zeros_like(
    other: Tensor,
    *,
    dtype: Any = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor of zeros with the same shape and device as `other`.

    Args:
        other (Tensor): The tensor to match shape and device from.
        dtype (Any): Override dtype. Defaults to None (use other's dtype).
        requires_grad (bool): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: A tensor of zeros.
    """
    # Use xp.zeros(shape) instead of xp.zeros_like(tensor) to avoid
    # triggering __array_function__ on the Tensor
    result: Tensor = xp.zeros(other.shape, dtype=dtype or other.dtype).view(Tensor)
    result.requires_grad = _GRAD_MODE_ENABLED and requires_grad
    return result


__all__ = [
    "copy_to_device",
    "ones_like",
    "zeros_like",
]
