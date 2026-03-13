"""All custom or extended (from numpy/cupy) operations on SADL Tensors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .backend import (
    TensorDevice,
    get_array_module_from_device,
    is_ndarray,
)
from .grad_mode import is_global_grad_mode_enabled
from .grad_ops import get_grad_op
from .tensor import Tensor, tensor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)


def _format_debug_arg(arg: Any) -> str:
    """Return a compact debug representation for an op argument.

    Args:
        arg (Any): Input argument to the op.

    Returns:
        str: Formatted string representation of the argument.
    """
    if is_ndarray(arg):
        device = TensorDevice.create(arg.device)
        return f"{type(arg).__name__}(shape={arg.shape}, dtype={arg.dtype}, device={device})"
    return f"{type(arg).__name__}({arg!r})"


def _determine_and_ensure_device(args: Iterable[Any]) -> TensorDevice:
    """Determine the single device on which an op with `args` should run.

    Simultaneously checks if all inputs to the op (`args`) are on the same device.

    Args:
        args (Iterable[Any]): The inputs to the op.

    Raises:
        RuntimeError: If inputs span multiple devices.

    Returns:
        TensorDevice: The device on which to run the op.
    """
    devices = {TensorDevice.create(arg.device) for arg in args if is_ndarray(arg)}
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
    if isinstance(data, Tensor):
        return data
    return tensor(data, device=device, requires_grad=False)


def dispatch_op(
    op_name: str,
    /,
    *,
    op_fn: Callable[[Any], Any],
    op_inputs: Iterable[Any],
    **kwargs: Any,
) -> Tensor:
    """Dispatches any array op to the right backend.

    Also includes the actual execution of the op.
    Additionally, if gradient tracking is enabled, a
    new node (represented by the result Tensor of the op)
    in a computation graph is created for backpropagation.

    Args:
        op_name (str): The name of the operation to dispatch and execute.
        op_fn (Callable[[Any], Any]): The callable function of the op.
        op_inputs (Iterable[Any]): The inputs to the op.
        **kwargs (Any): Additional kwargs to the op. A common example is `axis`.

    Raises:
        ValueError: If gradient racking is enabled but a backward
            function for the op is not supported.

    Returns:
        Tensor: The Tensor resulting from the op.
    """
    # ensure only scalars and arrays
    input_args = [a.data if isinstance(a, Tensor) else a for a in op_inputs]

    grad_tracking = is_global_grad_mode_enabled()

    device = _determine_and_ensure_device(input_args)

    backend = get_array_module_from_device(device)

    track_kwargs = kwargs.copy()

    logger.debug(
        'Op "%s": device=%s, backend=%s, inputs=[%s], kwargs=%s',
        op_name,
        device,
        getattr(backend, "__name__", type(backend).__name__),
        ", ".join(_format_debug_arg(arg) for arg in input_args),
        kwargs,
    )

    if op_name in ["max", "min"]:
        # execute the function, but retain the dimensions:
        keepdims_kwargs = kwargs.copy()
        keepdims_kwargs["keepdims"] = True
        result = op_fn(*input_args, **keepdims_kwargs)
        # create a mask for each axis, where True means the value in x
        #   at this position is an extremum (maximum or minimum, depending on "method"):
        x_mask = result == input_args[0]
        track_kwargs["x_mask"] = x_mask

        result = (
            result
            if kwargs.get("keepdims", False)
            else backend.squeeze(result, axis=kwargs.get("axis"))
        )
    elif op_name in ["maximum", "minimum"]:
        result = op_fn(*input_args, **kwargs)
        x_mask = result == input_args[0]
        track_kwargs["x_mask"] = x_mask
    else:
        result = op_fn(*input_args, **kwargs)

    result = backend.asarray(result)  # ensure array type for scalar reductions

    # Skip graph building when grad mode is disabled (e.g. during backward pass)
    if not grad_tracking:
        return Tensor(result, requires_grad=False)

    # ensure we always have Tensors in the graph:
    src = tuple(_to_tensor(i, device=device) for i in op_inputs)

    result_requires_grad = any(elem.requires_grad for elem in src)

    backward_fn = get_grad_op(op_name)

    result_tensor = Tensor(
        result,
        requires_grad=result_requires_grad,
    )

    if result_requires_grad:
        if backward_fn is None:
            raise ValueError(
                f'Operation "{op_name}" not supported, no backward function available.'
            )

        result_tensor.src = src
        result_tensor.backward_fn = backward_fn
        result_tensor.op_ctx = track_kwargs

    return result_tensor


__all__ = [
    "dispatch_op",
]
