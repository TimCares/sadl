"""Contains all operations that support gradient calculation.

Uses numpy as the backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from .backend import xp
from .utils import copy_array

if TYPE_CHECKING:
    from .tensor import Tensor


# Type alias for gradient operations (returns plain arrays, not Tensors)
# Gradients are raw numerical buffers without computation graph overhead
GradOp = Callable[..., tuple["xp.ndarray | None", ...]]

_GRAD_OPS_REGISTRY: dict[str, GradOp] = {}


def register_grad_op(func: GradOp) -> GradOp:
    """Decorator to register a grad operation to the global `GRAD_OPS_REGISTRY`.

    Expect `func` to have the naming convention `<operation>_backward`.
    `func` will then be registered under the key `<operation>`.

    Args:
        func (GradOp): The grad operation to register.

    Returns:
        GradOp: The original `func`, same as input.
    """
    operation_name = func.__name__.rsplit("_", maxsplit=1)[0]
    _GRAD_OPS_REGISTRY[operation_name] = func
    return func


def normalize_grad_op_name(
    *,
    name: str,
    is_reduce: bool = False,
) -> str:
    """Normalizes the name of the op for which the backward function is requested.

    Allows to use the same backward function for multiple ops.

    Args:
        name (str): The general op name.
        is_reduce (bool): Whether the . Defaults to False.

    Examples:
        Multiplication and division can have two
        designations::

            norm_name = _normalize_grad_op_name("multiply")
            >>> norm_name == "mul"
            norm_name = _normalize_grad_op_name("mul")
            >>> norm_name == "mul"

            norm_name = _normalize_grad_op_name("add", is_reduce=True)
            >>> norm_name == "sum"

            norm_name = _normalize_grad_op_name("add")
            >>> norm_name == "add"

    Returns:
        str: The normalized name of the operation.
    """
    if name == "multiply":
        return "mul"
    if name == "divide":
        return "div"
    if name == "add" and is_reduce:
        return "sum"
    return name


def get_grad_op(name: str) -> GradOp | None:
    """Gets the grad op by its forward name.

    `name="add"` to get the backward function for addition,
    `name="matmul"` for matrix multiplication, ...

    Args:
        name (str): Name of the forward op for which to get
            the grad function.

    Returns:
        GradOp | None: The grad op function, `None` if no
            backward/gradient function exists for the
            operation given by `name`.
    """
    return _GRAD_OPS_REGISTRY.get(normalize_grad_op_name(name=name))


def _broadcast_backward(
    x: Tensor,
    grad_out: xp.ndarray,
) -> xp.ndarray:
    """Applies a backward gradient operation on broadcasting.

    Effectively collapses `grad_out` by summing over all
    dimensions of `x` that were broadcasted.

    Args:
        x (Tensor): The Tensor that was broadcasted.
        grad_out (xp.ndarray): The gradient of the following operation.

    Returns:
        xp.ndarray: The computed gradient.
    """
    if x.shape == grad_out.shape:
        return grad_out  # shapes are the same, no broadcast happened

    collapse_dim: list[int] = []
    for i in range(max(x.ndim, grad_out.ndim)):
        idx_x = x.ndim - i - 1
        idx_grad_out = grad_out.ndim - i - 1
        if idx_x < 0 or x.shape[idx_x] < grad_out.shape[idx_grad_out]:
            collapse_dim.append(idx_grad_out)

    return xp.sum(grad_out, axis=tuple(collapse_dim), keepdims=True).reshape(x.shape)


def broadcastable(
    elem_wise_backward_fn: Callable[..., Any],
) -> Callable[..., Any]:
    """A decorator to extend element-wise backward gradient computing functions.

    This decorator adds the ability to support broadcasting.

    Args:
        elem_wise_backward_fn (Callable[[...], Any]): The backward
            function of an element-wise operation that should support
            broadcasting.

    Returns:
        Callable[[...], Any]: The wrapper function
        that supports broadcasting.
    """

    def wrapper(
        *inputs: Tensor,
        compute_grad: tuple[bool],
        grad_out: xp.ndarray,
        **kwargs: Any,
    ) -> tuple[xp.ndarray | None, xp.ndarray | None]:
        x, y = inputs
        grad_x, grad_y = elem_wise_backward_fn(
            *inputs,
            compute_grad=compute_grad,
            grad_out=grad_out,
            **kwargs,
        )
        grad_x = _broadcast_backward(x, grad_x) if grad_x is not None else None
        grad_y = _broadcast_backward(y, grad_y) if grad_y is not None else None
        return grad_x, grad_y

    # Trick so that register_grad_op decorator registers the,
    # now broadcastable backward function under the name of the
    # original backward function `elem_wise_backward_fn` (e.g. `add_backward`),
    # instead of under the name `wrapper`
    wrapper.__name__ = elem_wise_backward_fn.__name__

    return wrapper


@register_grad_op
def absolute_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None]:
    """Computes gradients for `abs(x) = |x| = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): Upstream gradient.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x = inputs[0]
    x_grad = xp.sign(x) * grad_out if compute_grad[0] else None
    return (x_grad,)


@register_grad_op
def negative_backward(
    *inputs: Tensor,  # noqa: ARG001
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None]:
    """Computes gradients for `-x = -1 * x  = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): Upstream gradient.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x_grad = -1 * grad_out if compute_grad[0] else None
    return (x_grad,)


@register_grad_op
@broadcastable
def add_backward(
    *inputs: Tensor,  # noqa: ARG001
    compute_grad: tuple[bool, bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Computes gradients for `x + y = z`.

    Args:
        *inputs (Tensor): Two inputs `(x, y)`.
        compute_grad (tuple[bool, bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): Upstream gradient.

    Returns:
        tuple[xp.ndarray | None, xp.ndarray | None]: Gradients for `(x, y)`, with
            `None` where `compute_grad[i]` is False.
    """
    x_grad = grad_out if compute_grad[0] else None
    y_grad = grad_out if compute_grad[1] else None
    return x_grad, y_grad


# Theoretically we do not need a backward for subtract
# -> "x - y = x + (-y) = z", so we could chain the
#   "negative" and "add" backward functions, but a single
#   "substract" function is more efficient
@register_grad_op
@broadcastable
def subtract_backward(
    *inputs: Tensor,  # noqa: ARG001
    compute_grad: tuple[bool, bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Computes gradients for `x - y = z`.

    Args:
        *inputs (Tensor): Two inputs `(x, y)`.
        compute_grad (tuple[bool, bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): Upstream gradient.

    Returns:
        tuple[xp.ndarray | None, xp.ndarray | None]: Gradients for `(x, y)`, with
            `None` where `compute_grad[i]` is False.
    """
    x_grad = grad_out if compute_grad[0] else None
    y_grad = -1 * grad_out if compute_grad[1] else None
    return x_grad, y_grad


@register_grad_op
@broadcastable
def mul_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool, bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Computes the gradient for multiplication `x * y = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 2.
            Expects: `tuple[0]` to be `x`, `tuple[1]` to be `y`.
        compute_grad (tuple[bool, bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None, xp.ndarray | None]: Gradients for `(x, y)`, with
            `None` where `compute_grad[i]` is False.
    """
    x, y = inputs
    grad_x = y * grad_out if compute_grad[0] else None
    grad_y = x * grad_out if compute_grad[1] else None
    return grad_x, grad_y


@register_grad_op
@broadcastable
def div_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool, bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Computes the gradient for division `x / y = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 2.
            Expects: `tuple[0]` to be `x`, `tuple[1]` to be `y`.
        compute_grad (tuple[bool, bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None, xp.ndarray | None]: Gradients for `(x, y)`, with
            `None` where `compute_grad[i]` is False.
    """
    x, y = inputs
    grad_x = grad_out / y if compute_grad[0] else None
    grad_y = -x * grad_out / (y * y) if compute_grad[1] else None
    return grad_x, grad_y


@register_grad_op
def matmul_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool, bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Computes the gradient for matrix multiplication `AB = Z`.

    Batched matrix multiplication is also supported, so inputs can
    have dimensions `(..., i, j)` and `(..., j, k)`

    Args:
        *inputs (Tensor): The inputs, expected to be of length 2.
            Expects: `tuple[0]` to be the first matrix (or batch of matrices),
            `tuple[1]` to be the second matrix (or batch of matrices).
        compute_grad (tuple[bool, bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None, xp.ndarray | None]: Gradients for `(A, B)`, with
            `None` where `compute_grad[i]` is False.
    """
    A, B = inputs
    grad_x = xp.matmul(grad_out, xp.swapaxes(B, -2, -1)) if compute_grad[0] else None
    grad_y = xp.matmul(xp.swapaxes(A, -2, -1), grad_out) if compute_grad[1] else None

    return grad_x, grad_y


@register_grad_op
def sqrt_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for square root `sqrt(x) = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x = inputs[0]
    grad_x = grad_out / (2 * xp.sqrt(x)) if compute_grad[0] else None
    return (grad_x,)


@register_grad_op
def power_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool, bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Computes the gradient for power `x^y = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 2.
            Expects: `tuple[0]` to be `x`, `tuple[1]` to be `y`.
        compute_grad (tuple[bool, bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None, xp.ndarray | None]: Gradients for `(x, y)`, with
            `None` where `compute_grad[i]` is False.
    """
    x, y = inputs
    grad_x = y * xp.pow(x, y - 1) * grad_out if compute_grad[0] else None
    grad_y = xp.pow(x, y) * xp.log(x) * grad_out if compute_grad[1] else None
    return grad_x, grad_y


@register_grad_op
def square_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for the square op (`x^2 = z`).

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x = inputs[0]
    grad_x = 2 * x * grad_out if compute_grad[0] else None
    return (grad_x,)


@register_grad_op
def exp_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for exponentiation `e^x = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x = inputs[0]
    grad_x = grad_out * xp.exp(x) if compute_grad[0] else None
    return (grad_x,)


@register_grad_op
def log_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for the logarithm `log(x) = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x = inputs[0]
    grad_x = grad_out / x if compute_grad[0] else None
    return (grad_x,)


@register_grad_op
def sin_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for the sinus function `sin(x) = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x = inputs[0]
    grad_x = xp.cos(x) * grad_out if compute_grad[0] else None
    return (grad_x,)


@register_grad_op
def cos_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for the cosine function `cos(x) = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x = inputs[0]
    grad_x = -xp.sin(x) * grad_out if compute_grad[0] else None
    return (grad_x,)


@register_grad_op
def sum_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
    **kwargs: Any,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for the sum function `sum(x) = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.
        **kwargs (Any): Additional arguments, expects `axis`
            to be present, denoting the axis of x over which the sum
            was done.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x = inputs[0]

    grad_x: xp.ndarray | None = None

    if compute_grad[0]:
        kwargs_axis = kwargs.get("axis")
        axis = tuple(kwargs_axis) if kwargs_axis is not None else tuple(range(x.ndim))

        grad_x = xp.broadcast_to(xp.expand_dims(grad_out, axis=axis), shape=x.shape)

    return (grad_x,)


@register_grad_op
def mean_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
    **kwargs: Any,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for the mean function `mean(x) = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.
        **kwargs (Any): Additional arguments, expects `axis`
            to be present, denoting the axis of x over which the sum
            was done.

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    x = inputs[0]

    grad_x: xp.ndarray | None = None

    if compute_grad[0]:
        kwargs_axis = kwargs.get("axis")
        axis = tuple(kwargs_axis) if kwargs_axis is not None else tuple(range(x.ndim))

        grad_x = xp.broadcast_to(xp.expand_dims(grad_out, axis=axis), shape=x.shape)

        n_reduced_elem = xp.prod([x.shape[a] for a in axis])

        grad_x = grad_x / n_reduced_elem

    return (grad_x,)


def _extremum_backward(
    *inputs: Tensor,
    op_type: Literal["min", "max"],
    grad_out: xp.ndarray,
    **kwargs: Any,
) -> tuple[xp.ndarray]:
    """Computes the gradient for an extremum function.

    Extemum is either `min` or `max`: `f(x) = z, f ∈ {min, max}`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        op_type (Literal["min", "max"]): The extremum type for which to
            compute the gradient. Must be in ["min", "max"].
        grad_out (xp.ndarray): The gradient of the following
            operation.
        **kwargs (Any): Additional arguments, tries to extract:

            - `axis`: Denoting the axis of x over which the
                operation was done (defaults to None).

            - `x_mask`: Stores whether a value in `x` is an extremum
                (`min` for minimum, `max` for maximum)
                along the reduced axes (must be of the same shape as `x`).
                This argument is **required** to avoid numerical instability
                when supressing all non-extrema values (this is the backward function).

            - `keepdims`: Whether the dimensions over which was
                reduced were retained (defaults to False).

    Returns:
        tuple[xp.ndarray]: The computed gradient with respect to `x`.
    """
    x = inputs[0]

    x_mask = kwargs.get("x_mask")
    if x_mask is None:
        raise ValueError(
            f'Missing keyword argument "x_mask" is required in backward for {op_type}.'
        )

    assert x_mask.shape == x.shape, "Extremum mask must have the same shape as x"

    kwargs_axis = kwargs.get("axis")
    if kwargs_axis is None:
        axis = tuple(range(x.ndim))
    else:
        axis = make_axis(ndim=x.ndim, axis_candidate=kwargs_axis)

    if not kwargs.get("keepdims", False):
        grad_out = xp.expand_dims(grad_out, axis=axis)

    grad_out = xp.broadcast_to(grad_out, shape=x.shape)

    count = xp.sum(x_mask, axis=axis, keepdims=True)
    assert xp.all(count > 0), (
        f'There must be at least one {op_type} along the reduced axis "{axis}"'
    )

    grad_x = xp.where(x_mask, grad_out / count, 0)
    # Scale the gradient by the inverse of the number of extremas there were
    #   along the reduced axes.
    # If one axis had 3 extremas, then the gradient is divided between them
    #   equally. In that case, the gradients for all 3 extremas are scaled by 1/3.

    return (grad_x,)


@register_grad_op
def max_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
    **kwargs: Any,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for the max function: max(x) = z.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.
        **kwargs (Any): Additional arguments, tries to extract:

            - `axis`: Denoting the axis of x over which the
                operation was done (defaults to None).

            - `x_mask`: Stores whether a value in `x` is a max
                along the reduced axes (must be of the same shape as `x`).
                This argument is **required** to avoid numerical instability
                when supressing all non-max values (this is the backward function).

            - `keepdims`: Whether the dimensions over which was
                reduced were retained (defaults to False).

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    if not compute_grad[0]:
        return (None,)
    # else ->
    return _extremum_backward(
        *inputs,
        op_type="max",
        grad_out=grad_out,
        **kwargs,
    )


@register_grad_op
def min_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
    **kwargs: Any,
) -> tuple[xp.ndarray | None]:
    """Computes the gradient for the min function: min(x) = z.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 1.
            Expects: `tuple[0]` to be `x`.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.
        **kwargs (Any): Additional arguments, tries to extract:

            - `axis`: Denoting the axis of x over which the
                operation was done (defaults to None).

            - `x_mask`: Stores whether a value in `x` is a min
                along the reduced axes (must be of the same shape as `x`).
                This argument is **required** to avoid numerical instability
                when supressing all non-min values (this is the backward function).

            - `keepdims`: Whether the dimensions over which was
                reduced were retained (defaults to False).

    Returns:
        tuple[xp.ndarray | None]: Gradient for `x`, or `None` if skipped.
    """
    if not compute_grad[0]:
        return (None,)
    # else ->
    return _extremum_backward(
        *inputs,
        op_type="min",
        grad_out=grad_out,
        **kwargs,
    )


def _element_wise_extremum_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool, bool],
    op_type: Literal["minimum", "maximum"],
    grad_out: xp.ndarray,
    **kwargs: Any,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Computes the gradient for an element-wise extremum function.

    Extemum is either `minimum` or `maximum`:
    `f(x, y) = z, f ∈ {minimum, maximum}`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 2.
            Expects: `tuple[0]` to be `x`, `tuple[1]` to be `y`.
        compute_grad (tuple[bool, bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        op_type (Literal["minimum", "maximum"]): The extremum type
            for which to compute the gradient.
            Must be in ["minimum", "maximum"].
        grad_out (xp.ndarray): The gradient of the following
            operation.
        **kwargs (Any): Additional arguments, tries to extract:

            - `x_mask`: Stores whether a value in `x` is an extremum
                (`min` for minimum, `max` for maximum) compared
                to the value at the same location in `y`
                (must be of the same shape as `x`). This argument
                is **required** to avoid numerical instability
                when supressing all non-extrema values (this is the backward function).

            - `where`: On which locations to apply `f`.

    Returns:
        tuple[xp.ndarray | None, xp.ndarray | None]: Gradients for `(x, y)`, with
            `None` where `compute_grad[i]` is False.
    """
    x = inputs[0]
    y = inputs[1]

    if x.shape != y.shape:
        raise ValueError(
            f"Both inputs must be of same shape, got {x.shape} for x and {y.shape} for y"
        )

    x_mask = kwargs.get("x_mask")
    if x_mask is None:
        raise ValueError(
            f'Missing keyword argument "x_mask" is required in backward for {op_type}.'
        )

    apply_grad_op = kwargs.get("where", 1)

    assert x_mask.shape == x.shape, "Extremum mask must have the same shape as x"

    both_extremum = x == y

    x_grad: xp.ndarray | None = None
    y_grad: xp.ndarray | None = None

    if compute_grad[0]:
        scale_x = xp.where(both_extremum, 0.5, x_mask)
        x_grad = apply_grad_op * scale_x * grad_out

    if compute_grad[1]:
        scale_y = xp.where(both_extremum, 0.5, 1 - x_mask)
        y_grad = apply_grad_op * scale_y * grad_out

    return x_grad, y_grad


@register_grad_op
@broadcastable
def maximum_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool, bool],
    grad_out: xp.ndarray,
    **kwargs: Any,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Computes the gradient for an element-wise maximum.

    Function is `maximum(x, y) = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 2.
            Expects: `tuple[0]` to be `x`, `tuple[1]` to be `y`.
        compute_grad (tuple[bool, bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.
        **kwargs (Any): Additional arguments, tries to extract:

            - `x_mask`: Stores whether a value in `x` is
                larger compared to the value at the same
                location in `y` (must be of the same shape as `x`).
                This argument is **required** to avoid numerical instability
                when supressing all non-maximum values (this is the backward function).

    Returns:
        tuple[xp.ndarray | None, xp.ndarray | None]: Gradients for `(x, y)`, with
            `None` where `compute_grad[i]` is False.
    """
    return _element_wise_extremum_backward(
        *inputs,
        compute_grad=compute_grad,
        op_type="maximum",
        grad_out=grad_out,
        **kwargs,
    )


@register_grad_op
@broadcastable
def minimum_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool, bool],
    grad_out: xp.ndarray,
    **kwargs: Any,
) -> tuple[xp.ndarray | None, xp.ndarray | None]:
    """Computes the gradient for an element-wise minimum.

    Function is `minimum(x, y) = z`.

    Args:
        *inputs (Tensor): The inputs, expected to be of length 2.
            Expects: `tuple[0]` to be `x`, `tuple[1]` to be `y`.
        compute_grad (tuple[bool, bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): The gradient of the following
            operation.
        **kwargs (Any): Additional arguments, tries to extract:

            - `x_mask`: Stores whether a value in `x` is
                smaller compared to the value at the same
                location in `y` (must be of the same shape as `x`).
                This argument is **required** to avoid numerical instability
                when supressing all non-minimum values (this is the backward function).

    Returns:
        tuple[xp.ndarray | None, xp.ndarray | None]: Gradients for `(x, y)`, with
            `None` where `compute_grad[i]` is False.
    """
    return _element_wise_extremum_backward(
        *inputs,
        compute_grad=compute_grad,
        op_type="minimum",
        grad_out=grad_out,
        **kwargs,
    )


@register_grad_op
def copy_to_device_backward(
    *inputs: Tensor,
    compute_grad: tuple[bool],
    grad_out: xp.ndarray,
) -> tuple[xp.ndarray | None]:
    """Computes gradients for copying a Tensor to a (different) device.

    Args:
        *inputs (Tensor): The input Tensor, should only be one, as
            a copy operation only operates on a single Tensor.
        compute_grad (tuple[bool]): Flags indicating which input gradients
            to compute, aligned with `inputs`.
        grad_out (xp.ndarray): Upstream gradient.

    Returns:
        tuple[xp.ndarray | None]: Gradients for the Tensor, with
            `None` where `compute_grad[i]` is False.
    """
    x = inputs[0]
    # Just pass through grad_out by reverting the "copy_to_device" op
    # That means copying grad_out back to the device of x
    x_grad = copy_array(array=grad_out, device=x.device) if compute_grad[0] else None
    return (x_grad,)


def make_axis(ndim: int, axis_candidate: Any) -> tuple[int, ...]:
    """Transforms the `axis` argument for numpy ops.

    Returns a consistent `tuple[int, ...]` type.
    Transforms negative axes into positive ones for
    consistency.

    Args:
        ndim (int): Number of dimensions of the array
            on which the numpy op is performed.
        axis_candidate (Any): The `axis` argument
            used in the numpy op.

    Raises:
        ValueError: If `axis_candidate` is an invalid
            numpy `axis` type.

    Returns:
        tuple[int, ...]: The consistent numpy `axis`
            argument.
    """
    if isinstance(axis_candidate, int):
        axis = (axis_candidate,)
    elif isinstance(axis_candidate, list) and all(isinstance(a, int) for a in axis_candidate):
        axis = tuple(axis_candidate)

    elif isinstance(axis_candidate, tuple) and all(isinstance(a, int) for a in axis_candidate):
        axis = axis_candidate

    else:
        raise ValueError(
            '"axis_candidate" must be in '
            f'[int, list[int], tuple[int]], found: "{type(axis_candidate).__name__}".'
        )

    def make_positive(value: int) -> int:
        if value < 0:
            return ndim + value  # normalize axes to positive values
        return value

    normalized_axis = tuple(map(make_positive, axis))

    assert all(0 <= a < ndim for a in normalized_axis)

    return normalized_axis


__all__ = [
    "GradOp",
    "get_grad_op",
    "register_grad_op",
]
