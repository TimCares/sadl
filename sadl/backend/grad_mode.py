"""Utility functionality for the global grad mode."""

import logging
from collections.abc import Callable
from types import TracebackType
from typing import ParamSpec, Self, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


_GRAD_MODE_ENABLED: bool = True


class no_grad:  # noqa: N801
    """Context manager to disable gradient tracking in the context."""

    def __enter__(self) -> Self:
        global _GRAD_MODE_ENABLED
        self.prev = _GRAD_MODE_ENABLED
        _GRAD_MODE_ENABLED = False
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        global _GRAD_MODE_ENABLED
        _GRAD_MODE_ENABLED = self.prev


def set_global_grad_mode(enabled: bool) -> None:
    """Sets the global grad mode to `enabled`.

    Args:
        enabled (bool): Whether to enable or disable
            gradient tracking.
    """
    global _GRAD_MODE_ENABLED
    _GRAD_MODE_ENABLED = enabled
    logger.debug(f"Gradient tracking {'enabled' if enabled else 'disabled'}")


def is_global_grad_mode_enabled() -> bool:
    """Gets the current global grad mode.

    Returns:
        bool: Whether gradient tracking is
            enabled or disabled.
    """
    return _GRAD_MODE_ENABLED


def no_grad_fn(fn: Callable[P, T]) -> Callable[P, T]:
    """Disables gradient tracking for all ops in the annotated function.

    This decorator preserves the original function's type signature.

    Args:
        fn (Callable[P, T]): The function in which to disable gradient tracking.

    Returns:
        The wrapped function with the same signature as the input.

    Example:
        >>> @no_grad_fn
        ... def inference(x: Tensor) -> Tensor:
        ...     return x * 2
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with no_grad():
            return fn(*args, **kwargs)

    return wrapper


__all__ = [
    "is_global_grad_mode_enabled",
    "no_grad",
    "no_grad_fn",
    "set_global_grad_mode",
]
