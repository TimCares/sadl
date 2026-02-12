from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from .backend import is_global_grad_mode_enabled, xp
from .tensor import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def traverse_attrs(
    root: object,
    target_type: type[T],
    on_target: Callable[[str, T], T | None],
    *,
    recurse_into: type | tuple[type, ...] = (),
) -> None:
    """Recursively traverse all attributes of **root** looking for instances of **target_type**.

    Walks every instance attribute (via `vars(root)`) and descends into
    lists, dicts, and objects whose type is listed in **recurse_into**.

    Args:
        root: The object whose attributes to traverse.
        target_type: The type to search for.
        on_target: Callback invoked for every match as `on_target(path, item)`.
            If it returns an instance, the original is replaced in-place.
            If it returns `None`, no replacement occurs.
        recurse_into: Type(s) whose `vars()` should be recursively
            traversed (in addition to the standard containers list and dict).

    Raises:
        TypeError: If a target or recurse-into instance is found inside a
            `set` or `tuple`.
    """
    # Normalise to tuple for isinstance checks
    _recurse_into: tuple[type, ...] = (
        (recurse_into,) if isinstance(recurse_into, type) else recurse_into
    )
    _forbidden: tuple[type, ...] = (target_type, *_recurse_into)

    def _handle(
        parent: Any,
        key: str | int,
        item: Any,
        path: str,
    ) -> None:
        if isinstance(item, target_type):
            result = on_target(path, item)
            if result is not None:
                # Replace the item in its parent container
                if isinstance(parent, dict):
                    parent[key] = result
                elif isinstance(parent, list):
                    parent[key] = result  # type: ignore[index]
                elif hasattr(parent, "__dict__") and isinstance(key, str):
                    setattr(parent, key, result)
                # Note: tuples are immutable, can't replace items in them
        elif isinstance(item, _recurse_into):
            # Recurse into nested objects of the recurse-into type(s)
            traverse_attrs(
                item,
                target_type,
                lambda p, t: on_target(f"{path}.{p}", t),
                recurse_into=_recurse_into,
            )
        elif isinstance(item, set):
            for elem in item:
                if isinstance(elem, _forbidden):
                    raise TypeError(
                        f'Found "{type(elem).__name__}" in property of type "set" '
                        f'for path "{path}". Sets are not allowed for storing '
                        f"these types (no stable ordering)."
                    )
        elif isinstance(item, tuple):
            for i, elem in enumerate(item):
                if isinstance(elem, _forbidden):
                    raise TypeError(
                        f'Found "{type(elem).__name__}" in property of type "tuple" '
                        f'for path "{path}[{i}]". Tuples are not allowed for storing '
                        f"these types (immutable, cannot replace). Use a list instead."
                    )
                # Still traverse nested structures (e.g., tuple containing a dict)
                _handle(item, i, elem, f"{path}[{i}]")
        elif isinstance(item, list):
            for i, elem in enumerate(item):
                _handle(item, i, elem, f"{path}[{i}]")
        elif isinstance(item, dict):
            for k, v in item.items():
                _handle(item, k, v, f"{path}{{{k}}}")

    for attr_name, attr_value in vars(root).items():
        _handle(root, attr_name, attr_value, attr_name)


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
    result.requires_grad = is_global_grad_mode_enabled() and requires_grad
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
    result.requires_grad = is_global_grad_mode_enabled() and requires_grad
    return result
