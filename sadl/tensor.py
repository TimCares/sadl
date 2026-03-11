"""Custom tensor implementations that support autograd."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from numpy.lib.mixins import NDArrayOperatorsMixin

from . import ops
from .backend import (
    DeviceLike,
    NDArray,
    NDArrayLike,
    TensorDevice,
    copy_array,
    is_ndarray,
)
from .grad_mode import is_global_grad_mode_enabled
from .grad_ops import normalize_grad_op_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from .grad_ops import GradOp


logger = logging.getLogger(__name__)


def _unwrap_tensor_index(key: Any) -> Any:
    """Recursively unwrap Tensor objects inside indexing keys.

    Args:
        key (Any): The key used to perform indexing.

    Returns:
        Any: Same key but with all Tensors unwrapped to NDArray.
    """
    if isinstance(key, Tensor):
        return key.data
    if isinstance(key, tuple):
        key = cast("tuple[Any, ...]", key)
        return tuple(_unwrap_tensor_index(elem) for elem in key)
    if isinstance(key, list):
        key = cast("list[Any]", key)
        return [_unwrap_tensor_index(elem) for elem in key]
    return key


class Tensor(NDArrayOperatorsMixin):
    """A tensor wrapper around arrays with autograd support."""

    def __init__(
        self,
        data: NDArray,
        *,
        requires_grad: bool = False,
        keep_grad: bool = False,
    ) -> None:
        """Initialize the Tensor.

        Args:
            data (NDArray): The backing array buffer.
            requires_grad (bool, optional): Whether to track gradients for this tensor.
                Defaults to False.
            keep_grad (bool, optional): Whether to retain the gradient after the backward
                pass. Should be True for leaf tensors like parameters. Defaults to False.
        """
        self.data = data

        self.requires_grad = is_global_grad_mode_enabled() and requires_grad

        self.grad: NDArray | None = None

        self.keep_grad = keep_grad

        # autodiff context:
        self.src: tuple[Tensor, ...] = ()
        self.backward_fn: GradOp | None = None
        self.op_ctx: dict[str, Any] = {}

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Method called when numpy/cupy ufuncs are called on Tensors, e.g. `np.sum`.

        Args:
            ufunc (np.ufunc): The numpy/cupy ufunc.
            method (str): Which type of ufunc this is. Important to detect reduction operations
                like `np.sum`.
            *inputs (Any): All inputs to the ufunc.
            **kwargs (Any): Additional kwargs to the ufunc. A common example is `axis`.

        Returns:
            Any: The Tensor resulting from the numpy/cupy operation.
        """
        op_name = normalize_grad_op_name(name=ufunc.__name__, is_reduce=method == "reduce")

        func = getattr(ufunc, method)

        from .dispatch import (  # noqa: PLC0415 (avoid circual import between tensor.py and dispatch.py)
            dispatch_op,
        )

        return dispatch_op(op_name, op_fn=func, op_inputs=inputs, **kwargs)

    def __array_function__(
        self,
        func: Any,
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Method called when numpy/cupy array functions are called on Tensors, e.g. `np.matmul`.

        Args:
            func (Any): The numpy/cupy array function.
            types (Iterable[type]): Types of the input args to the function. Ignored, types are
                detected from `args` at runtime.
            args (Iterable[Any]): All inputs to the array function.
            kwargs (Mapping[str, Any]): Additional kwargs to the ufunc. A common example is `axis`.

        Returns:
            Any: The Tensor resulting from the numpy/cupy operation.
        """
        op_name = normalize_grad_op_name(name=func.__name__)

        from .dispatch import (  # noqa: PLC0415 (avoid circual import between tensor.py and dispatch.py)
            dispatch_op,
        )

        return dispatch_op(op_name, op_fn=func, op_inputs=args, **kwargs)

    @property
    def device(self) -> TensorDevice:
        """Device the Tensor is on.

        Returns:
            TensorDevice: The device holding the backing buffer.
        """
        return TensorDevice.create(self.data.device)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data.

        Returns:
            tuple[int, ...]: Dimension sizes of the backing array.
        """
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the data.

        Returns:
            int: Number of axes in the backing array.
        """
        return len(self.shape)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Data type of the data.

        Returns:
            npt.DTypeLike: Element type of the backing array.
        """
        return self.data.dtype

    @property
    def size(self) -> int:
        """Number of elements in the data.

        Returns:
            int: Total number of elements across all dimensions.
        """
        return np.size(self.data)

    def astype(self, dtype: npt.DTypeLike) -> Tensor:
        """Return a copy of this Tensor cast to `dtype`.

        If `dtype` already matches, the original Tensor is returned as-is
        (no copy, no graph node). Otherwise a new Tensor is created whose
        underlying data has been cast. When gradient tracking is active and
        this Tensor is a non-leaf node in the computation graph, the cast is
        recorded so that gradients flow back through it (by casting the
        upstream gradient back to the original dtype).

        Note: This operation focuses on the underlying data, not the
        computation graph. It does not accept an `in_place` flag: Mutating
        dtype in-place on a non-leaf tensor would silently corrupt any
        subsequent backward pass.

        Args:
            dtype (npt.DTypeLike): The target dtype.

        Returns:
            Tensor: A Tensor with the requested dtype. Returns `self` if the
                dtype is already correct.
        """
        return ops.astype(self, dtype=dtype)

    def is_leaf(self) -> bool:
        """Whether this Tensor is a leaf in a computation graph.

        Checks whether it has no src/parents from which it was created.

        Returns:
            bool: If it is a leaf (`True`), or not (`False`).
        """
        return len(self.src) == 0

    def copy_to_device(self, device: TensorDevice) -> Tensor:
        """Copy tensor data to `device`.

        For intermediate tensors in a computation graph (non-leaf with sources
        that require grad), this is a tracked operation so gradients flow back.
        For leaf tensors, this is a utility operation.

        Note: If the Tensor already is on `device`, no copy is created. Instead,
        the Tensor is returned as is.

        Args:
            device (TensorDevice): The device to copy to. Should either
                be `cpu` or an integer specifying the GPU id.

        Returns:
            Tensor: A tensor with the same data, now on `device`.
        """
        return ops.copy_to_device(self, device=device)

    def detach(
        self,
        *,
        in_place: bool = False,
    ) -> Tensor:
        """Detatch the Tensor from the computation graph.

        If `in_place` is `False`, the resulting Tensor is a deep copy
        without any graph context and therefore a leaf.

        Note: This behaves different from pytorch, where `requires_grad`
        ios set to `False`. This version **retains** `requires_grad`,
        so if `requires_grad=True` and `in_place=False`, the new Tensor
        can be used as a fresh leaf in a new computation graph.

        However, the gradient is always removed.

        Args:
            in_place (bool): Whether to detach the current Tensor
                in-place (`True`), which would cut the current computation
                graph on that node, or to detach a copy (including the
                memory buffer, unlike in Pytorch) of the
                current Tensor (`False`), which does **not**
                break the current computation graph. Defaults to False.

        Returns:
            Tensor: The resulting Tensor. If `in_place` is `True`, it will
                be the same one identity-wise.
        """
        if in_place:
            self.grad = None
            self.src = ()
            self.backward_fn = None
            self.op_ctx = {}
            return self

        # becomes a leaf
        return tensor(
            self.data,
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            keep_grad=self.keep_grad,
        )

    def cpu(self) -> Tensor:
        """Move the Tensor to the cpu.

        Note: If the Tensor already is on the cpu,
        no copy is created. Instead, the Tensor is returned as is.

        Returns:
            Tensor: A **copy** of the Tensor on the cpu,
                if it wasn't on the cpu before.
        """
        return self.copy_to_device(device=TensorDevice("cpu"))

    def gpu(self, device_id: int = 0) -> Tensor:
        """Move the Tensor to a gpu with `id`.

        Note: If the Tensor already is on the gpu `device_id`,
        no copy is created. Instead, the Tensor is returned as is.

        Args:
            device_id (int): The id of the gpu to which the Tensor
                should be copied. Defaults to 0.

        Returns:
            Tensor: A **copy** of the Tensor on the specified gpu,
                if it wasn't on the gpu `device_id` before.
        """
        return self.copy_to_device(device=TensorDevice("cuda", device_id=device_id))

    def item(self) -> Any:
        """Convert Tensor to a numeric scalar.

        Returns:
            Any: The scalar.
        """
        return self.data.item()

    def __len__(self) -> int:
        """Length of the first dimension.

        Returns:
            int: Size of axis 0.
        """
        return self.shape[0]

    def __getitem__(self, key: Any) -> Any:
        """Index into the tensor.

        Args:
            key (Any): An index, slice, integer array, or tuple thereof.
                Tensor indices are automatically unwrapped to their underlying arrays.

        Returns:
            Any: The selected sub-data. If not a scalar, a Tensor with `required_grad=False`
                is returned, as index access is non-differentiable.
        """
        result = self.data[_unwrap_tensor_index(key)]

        if not is_ndarray(result):
            return result

        return Tensor(result)

    def __setitem__(self, key: Any, value: Tensor | NDArrayLike) -> None:
        """Set values in the tensor by index.

        Args:
            key (Any): An index, slice, integer array, or tuple thereof.
                Tensor indices are automatically unwrapped to their underlying arrays.
            value (Tensor | NDArrayLike): Values to assign at the indexed positions.
        """
        self.data[_unwrap_tensor_index(key)] = value.data if isinstance(value, Tensor) else value

    def __hash__(self) -> int:
        """Identity-based hash for use in sets/dicts (computation graph tracking).

        Returns:
            int: Object identity as hash value.
        """
        return id(self)

    def __str__(self) -> str:
        """Human-readable string representation of the tensor data.

        Returns:
            str: String of the form `Tensor(<data>)`.
        """
        class_name = self.__class__.__name__
        prefix = f"{class_name}("
        data_str = str(self.data)
        # Indent continuation lines so they align with the opening bracket,
        # matching the style numpy/PyTorch use for multi-dimensional arrays.
        indented = data_str.replace("\n", "\n" + " " * len(prefix))
        return f"{prefix}{indented})"

    def __repr__(self) -> str:
        """Detailed string representation including dtype, device, and requires_grad.

        Returns:
            str: String of the form
                `Tensor(<data>, dtype=..., device='...', requires_grad=...)`.
        """
        class_name = self.__class__.__name__
        prefix = f"{class_name}("
        data_str = str(self.data)
        indented = data_str.replace("\n", "\n" + " " * len(prefix))

        device_str = (
            self.device.type
            if self.device.type == "cpu"
            else f"{self.device.type}:{self.device.device_id}"
        )

        return (
            f"{prefix}{indented}, "
            f"dtype={self.dtype!r}, "
            f"device='{device_str}', "
            f"requires_grad={self.requires_grad})"
        )


class Parameter(Tensor):
    """A special Tensor that should be part of a model to optimize.

    Parameters have an additional `is_training` attribute for controlling
    behavior of layers like Dropout and BatchNorm.
    """

    def __init__(
        self,
        data: Tensor | NDArray,
        *,
        is_training: bool = True,
    ) -> None:
        """Initialize the Parameter.

        Args:
            data (Tensor | NDArray): The initial parameter data. If a Tensor is passed,
                its underlying array is used.
            is_training (bool, optional): Whether the parameter is in training mode,
                relevant for layers like Dropout or BatchNorm. Defaults to True.
        """
        if isinstance(data, Tensor):
            data = data.data

        if np.issubdtype(data.dtype, np.integer):
            raise ValueError("Parameter must have float type, found int.")

        super().__init__(
            data,
            requires_grad=True,
            keep_grad=True,
        )
        self.is_training = is_training

    def copy_to_device(self, device: TensorDevice) -> Parameter:
        """Copy parameter data to `device`.

        This is a utility operation (not tracked in the
        computation graph). Consequently, the
        **copy-semantic is different to normal Tensors**.

        Note: If the Parameter already is on `device`, no copy is created.
        Instead, the Parameter is returned as is.

        Args:
            device (TensorDevice): The device to copy to. Should either
                be `cpu` or an integer specifying the GPU id.

        Returns:
            Parameter: A Parameter with the same data, now on `device`.
        """
        if self.device == device:
            return self

        new_data = copy_array(self.data, device)
        return Parameter(new_data, is_training=self.is_training)


def tensor(
    data: Tensor | NDArrayLike,
    *,
    device: DeviceLike | None = None,
    dtype: npt.DTypeLike | None = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Factory function to create a Tensor on the specified device with the specified dtype.

    This function always creates a new buffer in memory.

    Args:
        data (Tensor | NDArrayLike): The data from which to create the Tensor
            (can be numeric scalar, list, array, and even a Tensor).
            If Tensor, this function acts as a copy operation (without grad tracking).
        device (TensorDevice | None, optional): The device on which the Tensor
            should be created. Defaults to None, which infers TensorDevice("cpu").
        dtype (npt.DTypeLike | None, optional): The data type of the array data.
            Defaults to None, meaning dtype is inferred from data.
        requires_grad (bool, optional): Whether to track gradients. Defaults to False.
        keep_grad (bool, optional): Whether to retain gradients after backward. Defaults to False.

    Returns:
        Tensor: The created Tensor.
    """
    if isinstance(data, Tensor):
        data = data.data

    if device is None:
        device = TensorDevice("cpu")
    device = TensorDevice.create(device)

    data = copy_array(data, device=device, dtype=dtype)

    return Tensor(
        data,
        requires_grad=requires_grad,
        keep_grad=keep_grad,
    )


def ones(
    shape: tuple[int, ...],
    *,
    dtype: npt.DTypeLike | None = None,
    device: DeviceLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor filled with ones.

    Args:
        shape (tuple[int, ...]): Shape of the resulting Tensor.
        dtype (npt.DTypeLike | None, optional): Data type. If None, will be np.float32. Defaults to None.
        device (DeviceLike | None, optional): Target device. If None, will be CPU. Defaults to None.
        requires_grad (bool, optional): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: A Tensor of ones.
    """
    return tensor(
        np.ones(shape),
        dtype=dtype or np.float32,
        device=device,
        requires_grad=requires_grad,
    )


def zeros(
    shape: tuple[int, ...],
    *,
    dtype: npt.DTypeLike | None = None,
    device: DeviceLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor filled with zeros.

    Args:
        shape (tuple[int, ...]): Shape of the resulting Tensor.
        dtype (npt.DTypeLike | None, optional): Data type. If None, will be np.float32. Defaults to None.
        device (DeviceLike | None, optional): Target device. If None, will be CPU. Defaults to None.
        requires_grad (bool, optional): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: A Tensor of zeros.
    """
    return tensor(
        np.zeros(shape),
        dtype=dtype or np.float32,
        device=device,
        requires_grad=requires_grad,
    )


def eye(
    n: int,
    m: int | None = None,
    *,
    dtype: npt.DTypeLike | None = None,
    device: DeviceLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a 2D Tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): Number of rows.
        m (int | None, optional): Number of columns. If None, uses `n`.
            Defaults to None.
        dtype (npt.DTypeLike | None, optional): Data type. If None, will be np.float32.
            Defaults to None.
        device (DeviceLike | None, optional): Target device. If None, will be CPU.
            Defaults to None.
        requires_grad (bool, optional): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: An identity-like Tensor.
    """
    return tensor(
        np.eye(n, m),
        dtype=dtype or np.float32,
        device=device,
        requires_grad=requires_grad,
    )


def ones_like(
    other: Tensor,
    *,
    dtype: npt.DTypeLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor of ones matching the shape and device of `other`.

    Args:
        other (Tensor): Reference tensor for shape and device.
        dtype (npt.DTypeLike | None, optional): Override dtype. If None, dtype will
            be determined from `other`. Defaults to None.
        requires_grad (bool, optional): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: A Tensor of ones.
    """
    return ones(
        other.shape,
        dtype=dtype if dtype is not None else other.dtype,
        device=other.device,
        requires_grad=requires_grad,
    )


def zeros_like(
    other: Tensor,
    *,
    dtype: npt.DTypeLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor of zeros matching the shape and device of `other`.

    Args:
        other (Tensor): Reference tensor for shape and device.
        dtype (npt.DTypeLike | None, optional): Override dtype. If None, dtype will
            be determined from `other`. Defaults to None.
        requires_grad (bool, optional): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: A Tensor of zeros.
    """
    return zeros(
        other.shape,
        dtype=dtype if dtype is not None else other.dtype,
        device=other.device,
        requires_grad=requires_grad,
    )


__all__ = [
    "Parameter",
    "Tensor",
    "eye",
    "ones",
    "ones_like",
    "tensor",
    "zeros",
    "zeros_like",
]
