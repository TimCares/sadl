"""All custom or extended (from numpy/cupy) operations on SADL Tensors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from .backend import (
    BACKEND,
    ArrayModule,
    NDArray,
    NDArrayLike,
    TensorDevice,
    TensorDType,
    get_array_module_from_device,
)
from .grad_mode import is_global_grad_mode_enabled
from .grad_ops import (
    GradOp,
    absolute_backward,
    add_backward,
    astype_backward,
    copy_to_device_backward,
    cos_backward,
    div_backward,
    exp_backward,
    log_backward,
    matmul_backward,
    max_backward,
    maximum_backward,
    mean_backward,
    min_backward,
    minimum_backward,
    mul_backward,
    negative_backward,
    power_backward,
    reshape_backward,
    sin_backward,
    sqrt_backward,
    square_backward,
    subtract_backward,
    sum_backward,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .tensor import Tensor


logger = logging.getLogger(__name__)


def _format_debug_arg(arg: Any) -> str:
    """Return a compact debug representation for an op argument.

    Args:
        arg (Any): Input argument to the op.

    Returns:
        str: Formatted string representation of the argument.
    """
    if isinstance(arg, NDArray):
        arg = cast("NDArray", arg)
        device = TensorDevice.create(getattr(arg, "device", "cpu"))
        return f"{type(arg).__name__}(shape={arg.shape}, dtype={arg.dtype}, device={device})"
    return f"{type(arg).__name__}({arg!r})"


def _determine_device(args: list[NDArrayLike]) -> TensorDevice:
    """Determine the single device on which an op with `args` should run.

    Args:
        args (list[NDArrayLike]): The inputs to the op.

    Raises:
        RuntimeError: If inputs span multiple devices.

    Returns:
        TensorDevice: The device on which to run the op.
    """
    devices = {
        TensorDevice.create(getattr(arg, "device", "cpu"))
        for arg in args
        if isinstance(arg, NDArray)
    }
    if len(devices) > 1:
        raise RuntimeError(f"All inputs must be on the same device, found {devices}")
    return next(iter(devices)) if devices else TensorDevice("cpu")


def _to_tensor(data: Any, device: TensorDevice) -> Tensor:
    """Convert input to Tensor.

    Non-Tensors become Tensors with requires_grad=False.

    Creates a new memory buffer, so a copy of `data`.

    Args:
        data (Any): The data to convert to a Tensor.
        device (TensorDevice): The device on which to create the Tensor.

    Returns:
        Tensor: The Tensor.
    """
    from .tensor import Tensor, tensor  # noqa: PLC0415

    if isinstance(data, Tensor):
        # do not copy to "device" -> used in the wrapper below,
        # which requires all inputs to an op to be on the same device
        return data
    return tensor(data, device=device, requires_grad=False)


def sadl_op(
    backward_fn: GradOp,
) -> Callable[[Callable[..., NDArray]], Callable[..., Tensor]]:
    """A decorator to extend sadl ops with automatic tracking for backpropagation.

    The decorator extracts raw arrays from any `Tensor` arguments, resolves
    the correct array library (numpy or cupy) via `get_array_module`, and
    injects it into the wrapped function as the `__backend__` keyword
    argument. Each op uses `__backend__.sum(...)` etc. instead of relying
    on the global `xp`, so it is always dispatched to the library that owns
    the memory.

    Args:
        backward_fn (GradOp): The backward function of the decorated op.

    Returns:
        Callable: A decorator that wraps the op function with automatic
            graph tracking when gradient tracking is enabled.
    """

    def decorator(op_fn: Callable[..., NDArray]) -> Callable[..., Tensor]:
        def wrapper(*args: Tensor | NDArrayLike, **kwargs: Any) -> Tensor:
            from .tensor import Tensor  # noqa: PLC0415

            # ensure only scalars and arrays
            input_args = [a.data if isinstance(a, Tensor) else a for a in args]

            device = _determine_device(input_args)

            backend = get_array_module_from_device(device)

            track_kwargs = kwargs.copy()

            logger.debug(
                'Op "%s": device=%s, backend=%s, inputs=[%s], kwargs=%s',
                op_fn.__name__,
                device,
                getattr(backend, "__name__", type(backend).__name__),
                ", ".join(_format_debug_arg(arg) for arg in input_args),
                kwargs,
            )

            if op_fn.__name__ in ["max", "min"]:
                # execute the function, but retain the dimensions:
                keepdims_kwargs = kwargs.copy()
                keepdims_kwargs["keepdims"] = True
                result = op_fn(*input_args, __backend__=backend, **keepdims_kwargs)
                # create a mask for each axis, where True means the value in x
                #   at this position is an extremum (maximum or minimum, depending on "method"):
                x_mask = result == input_args[0]
                track_kwargs["x_mask"] = x_mask

                result = (
                    result
                    if kwargs.get("keepdims", False)
                    else backend.squeeze(result, axis=kwargs.get("axis"))
                )
            elif op_fn.__name__ in ["maximum", "minimum"]:
                result = op_fn(*input_args, __backend__=backend, **kwargs)
                x_mask = result == input_args[0]
                track_kwargs["x_mask"] = x_mask
            else:
                result = op_fn(*input_args, __backend__=backend, **kwargs)

            result = backend.asarray(result)  # ensure array type for scalar reductions

            # Skip graph building when grad mode is disabled (e.g. during backward pass)
            if not is_global_grad_mode_enabled():
                return Tensor(result, requires_grad=False)

            # ensure we always have Tensors in the graph:
            src = tuple(_to_tensor(i, device=device) for i in args)

            result_tensor = Tensor(
                result,
                requires_grad=any(elem.requires_grad for elem in src),
            )
            result_tensor.src = src
            result_tensor.backward_fn = backward_fn
            result_tensor.op_ctx = track_kwargs

            return result_tensor

        return wrapper

    return decorator


# =============================================================================
# Unary ops
# =============================================================================


@sadl_op(backward_fn=absolute_backward)
def absolute(x: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise absolute value.

    Args:
        x (NDArrayLike): Input array.

    Returns:
        Tensor: Element-wise |x|.
    """
    return __backend__.absolute(x)


@sadl_op(backward_fn=negative_backward)
def negative(x: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise numerical negative.

    Args:
        x (NDArrayLike): Input array.

    Returns:
        Tensor: Element-wise -x.
    """
    return __backend__.negative(x)


@sadl_op(backward_fn=sqrt_backward)
def sqrt(x: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise square root.

    Args:
        x (NDArrayLike): Input array. Must be non-negative.

    Returns:
        Tensor: Element-wise sqrt(x).
    """
    return __backend__.sqrt(x)


@sadl_op(backward_fn=square_backward)
def square(x: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise square.

    Args:
        x (NDArrayLike): Input array.

    Returns:
        Tensor: Element-wise x ** 2.
    """
    return __backend__.square(x)


@sadl_op(backward_fn=exp_backward)
def exp(x: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise natural exponential.

    Args:
        x (NDArrayLike): Input array.

    Returns:
        Tensor: Element-wise e ** x.
    """
    return __backend__.exp(x)


@sadl_op(backward_fn=log_backward)
def log(x: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise natural logarithm.

    Args:
        x (NDArrayLike): Input array. Must be positive.

    Returns:
        Tensor: Element-wise ln(x).
    """
    return __backend__.log(x)


@sadl_op(backward_fn=sin_backward)
def sin(x: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise sine.

    Args:
        x (NDArrayLike): Input array, in radians.

    Returns:
        Tensor: Element-wise sin(x).
    """
    return __backend__.sin(x)


@sadl_op(backward_fn=cos_backward)
def cos(x: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise cosine.

    Args:
        x (NDArrayLike): Input array, in radians.

    Returns:
        Tensor: Element-wise cos(x).
    """
    return __backend__.cos(x)


# =============================================================================
# Binary element-wise ops
# =============================================================================


@sadl_op(backward_fn=add_backward)
def add(x1: NDArrayLike, x2: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise addition.

    Args:
        x1 (NDArrayLike): First operand.
        x2 (NDArrayLike): Second operand.

    Returns:
        Tensor: Element-wise x1 + x2.
    """
    return __backend__.add(x1, x2)


@sadl_op(backward_fn=subtract_backward)
def subtract(x1: NDArrayLike, x2: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise subtraction.

    Args:
        x1 (NDArrayLike): Minuend.
        x2 (NDArrayLike): Subtrahend.

    Returns:
        Tensor: Element-wise x1 - x2.
    """
    return __backend__.subtract(x1, x2)


@sadl_op(backward_fn=mul_backward)
def multiply(x1: NDArrayLike, x2: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise multiplication.

    Args:
        x1 (NDArrayLike): First operand.
        x2 (NDArrayLike): Second operand.

    Returns:
        Tensor: Element-wise x1 * x2.
    """
    return __backend__.multiply(x1, x2)


@sadl_op(backward_fn=div_backward)
def divide(x1: NDArrayLike, x2: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise division.

    Args:
        x1 (NDArrayLike): Numerator.
        x2 (NDArrayLike): Denominator.

    Returns:
        Tensor: Element-wise x1 / x2.
    """
    return __backend__.divide(x1, x2)


@sadl_op(backward_fn=power_backward)
def power(x1: NDArrayLike, x2: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise exponentiation: x1 ** x2.

    Args:
        x1 (NDArrayLike): Base.
        x2 (NDArrayLike): Exponent.

    Returns:
        Tensor: Element-wise x1 ** x2.
    """
    return __backend__.power(x1, x2)


@sadl_op(backward_fn=maximum_backward)
def maximum(x1: NDArrayLike, x2: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise maximum of two arrays.

    Args:
        x1 (NDArrayLike): First operand.
        x2 (NDArrayLike): Second operand.

    Returns:
        Tensor: Element-wise max(x1, x2).
    """
    return __backend__.maximum(x1, x2)


@sadl_op(backward_fn=minimum_backward)
def minimum(x1: NDArrayLike, x2: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Element-wise minimum of two arrays.

    Args:
        x1 (NDArrayLike): First operand.
        x2 (NDArrayLike): Second operand.

    Returns:
        Tensor: Element-wise min(x1, x2).
    """
    return __backend__.minimum(x1, x2)


@sadl_op(backward_fn=matmul_backward)
def matmul(x1: NDArrayLike, x2: NDArrayLike, /, *, __backend__: ArrayModule) -> NDArray:
    """Matrix multiplication (supports batched inputs).

    Args:
        x1 (NDArrayLike): Left-hand matrix or batch of matrices.
        x2 (NDArrayLike): Right-hand matrix or batch of matrices.

    Returns:
        Tensor: Result of x1 @ x2.
    """
    return __backend__.matmul(x1, x2)


# =============================================================================
# Reduction ops
# =============================================================================


@sadl_op(backward_fn=sum_backward)
def sum(
    x: NDArrayLike,
    /,
    *,
    __backend__: ArrayModule,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> NDArray:
    """Sum of array elements along a given axis.

    Args:
        x (NDArrayLike): Input array.
        axis (int | tuple[int, ...] | list[int] | None, optional): Axis or axes to reduce.
            If None, all elements are summed. Defaults to None.
        keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
            Defaults to False.

    Returns:
        Tensor: Sum of elements along the specified axis.
    """
    return __backend__.sum(x, axis=axis, keepdims=keepdims)


@sadl_op(backward_fn=mean_backward)
def mean(
    x: NDArrayLike,
    /,
    *,
    __backend__: ArrayModule,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> NDArray:
    """Arithmetic mean along a given axis.

    Args:
        x (NDArrayLike): Input array.
        axis (int | tuple[int, ...] | list[int] | None, optional): Axis or axes to reduce.
            If None, the mean of all elements is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
            Defaults to False.

    Returns:
        Tensor: Mean of elements along the specified axis.
    """
    return __backend__.mean(x, axis=axis, keepdims=keepdims)


@sadl_op(backward_fn=max_backward)
def max(
    x: NDArrayLike,
    /,
    *,
    __backend__: ArrayModule,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> NDArray:
    """Maximum along a given axis.

    Args:
        x (NDArrayLike): Input array.
        axis (int | tuple[int, ...] | list[int] | None, optional): Axis or axes to reduce.
            If None, the maximum over all elements is returned. Defaults to None.
        keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
            Defaults to False.

    Returns:
        Tensor: Maximum value along the specified axis.
    """
    return __backend__.max(x, axis=axis, keepdims=keepdims)


@sadl_op(backward_fn=min_backward)
def min(
    x: NDArrayLike,
    /,
    *,
    __backend__: ArrayModule,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> NDArray:
    """Minimum along a given axis.

    Args:
        x (NDArrayLike): Input array.
        axis (int | tuple[int, ...] | list[int] | None, optional): Axis or axes to reduce.
            If None, the minimum over all elements is returned. Defaults to None.
        keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
            Defaults to False.

    Returns:
        Tensor: Minimum value along the specified axis.
    """
    return __backend__.min(x, axis=axis, keepdims=keepdims)


# =============================================================================
# Shape ops
# =============================================================================


@sadl_op(backward_fn=reshape_backward)
def reshape(x: NDArrayLike, /, *, __backend__: ArrayModule, shape: tuple[int, ...]) -> NDArray:
    """Reshape an array to the given shape.

    Args:
        x (NDArrayLike): Input array.
        shape (tuple[int, ...]): Target shape. Must be compatible with the total number
            of elements in `x`.

    Returns:
        Tensor: Array with the same data, viewed under the new shape.
    """
    return __backend__.reshape(x, shape)


# =============================================================================
# Non-differentiable reduction / index ops
# =============================================================================


def argmax(
    x: Tensor | NDArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Indices of the maximum values along a given axis.

    This operation is not differentiable.

    Args:
        x (Tensor | NDArrayLike): Input array.
        axis (int | tuple[int, ...] | list[int] | None, optional): Axis along which to
            find the maximum. If None, operates on the flattened array. Defaults to None.
        keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
            Defaults to False.

    Returns:
        Tensor: Integer indices of the maximum values (requires_grad=False).
    """
    from .tensor import Tensor  # noqa: PLC0415

    input_arg = x.data if isinstance(x, Tensor) else x
    device = _determine_device([input_arg])
    backend = get_array_module_from_device(device)
    result = backend.asarray(backend.argmax(input_arg, axis=axis, keepdims=keepdims))
    return Tensor(result, requires_grad=False)


def argmin(
    x: Tensor | NDArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | list[int] | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Indices of the minimum values along a given axis.

    This operation is not differentiable.

    Args:
        x (Tensor | NDArrayLike): Input array.
        axis (int | tuple[int, ...] | list[int] | None, optional): Axis along which to
            find the minimum. If None, operates on the flattened array. Defaults to None.
        keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
            Defaults to False.

    Returns:
        Tensor: Integer indices of the minimum values (requires_grad=False).
    """
    from .tensor import Tensor  # noqa: PLC0415

    input_arg = x.data if isinstance(x, Tensor) else x
    device = _determine_device([input_arg])
    backend = get_array_module_from_device(device)
    result = backend.asarray(backend.argmin(input_arg, axis=axis, keepdims=keepdims))
    return Tensor(result, requires_grad=False)


# =============================================================================
# Tensor-level ops (not wrapped by sadl_op, they build their own graph nodes)
# =============================================================================


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


def astype(x: Tensor, /, *, dtype: TensorDType) -> Tensor:
    """Create a copy of a Tensor `x` with its dtype given by `dtype`.

    Args:
        x (Tensor): The Tensor to have as `dtype`.
        dtype (TensorDType): The dtype.

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
    "absolute",
    "add",
    "argmax",
    "argmin",
    "astype",
    "copy_to_device",
    "cos",
    "divide",
    "exp",
    "log",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "multiply",
    "negative",
    "power",
    "reshape",
    "sadl_op",
    "sin",
    "sqrt",
    "square",
    "subtract",
    "sum",
]
