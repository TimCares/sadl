"""Custom tensor implementations that support autograd."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .backend import (
    ArrayModule,
    DeviceLike,
    NDArray,
    NDArrayLike,
    TensorDevice,
    TensorDType,
    copy_array,
    get_array_module,
)
from .backend.dtype import Float32
from .grad_mode import is_global_grad_mode_enabled

if TYPE_CHECKING:
    from . import ops as _ops_mod
    from .grad_ops import GradOp
else:

    class _OpsFacade:
        """Deferred proxy for the ops module (avoids tensor <-> ops import cycle)."""

        def __getattr__(self, name: str) -> Any:
            from . import ops  # noqa: PLC0415

            value = getattr(ops, name)
            setattr(self, name, value)
            return value

    _ops_mod = _OpsFacade()

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


def _ensure_comparable(x: Tensor, y: object) -> None:
    """Validate that a comparison target is compatible with `x`'s device.

    Args:
        x (Tensor): Tensor that is used in the comparison.
        y (object): Object the Tensor is compared to.

    Raises:
        RuntimeError: If `y` is not on the same device as `x`.
    """
    if isinstance(y, Tensor):
        other_device = y.device
    elif isinstance(y, NDArray):
        y = cast("NDArray", y)
        other_device = TensorDevice.create(getattr(y, "device", "cpu"))
    else:
        return

    if x.device != other_device:
        raise RuntimeError(
            f"Tensor comparison requires matching devices, found {x.device} and {other_device}."
        )


class Tensor:
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

    @property
    def data(self) -> NDArray:
        """The backing ndarray buffer.

        Returns:
            NDArray: The raw array owned by this Tensor.
        """
        return self._data

    @data.setter
    def data(self, value: NDArray) -> None:
        """Set the backing buffer and refresh the owning array module cache.

        Args:
            value (NDArray): New array buffer to assign.
        """
        if not isinstance(value, NDArray):
            raise TypeError(
                "Tensor class expects an array as data, "
                "If you would like to provide a list or scalar "
                "use the 'tensor' factory function instead."
            )
        self._data = value
        self._xp = get_array_module(value)

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
    def dtype(self) -> TensorDType:
        """Data type of the data.

        Returns:
            TensorDType: Element type of the backing array.
        """
        return TensorDType.from_backend(self.data.dtype)

    @property
    def size(self) -> int:
        """Number of elements in the data.

        Returns:
            int: Total number of elements across all dimensions.
        """
        return int(self._xp.size(self.data))

    @property
    def array_module(self) -> ArrayModule:
        """The array library (numpy or cupy) that owns the `data` buffer.

        Returns:
            ArrayModule: The active array module for this Tensor.
        """
        return self._xp

    def astype(self, dtype: TensorDType) -> Tensor:
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
            dtype (TensorDType): The target dtype.

        Returns:
            Tensor: A Tensor with the requested dtype. Returns `self` if the
                dtype is already correct.
        """
        return _ops_mod.astype(self, dtype=dtype)

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
        return _ops_mod.copy_to_device(self, device=device)

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

    # ------------------------------------------------------------------
    # Arithmetic dunder methods
    # ------------------------------------------------------------------

    def __add__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise addition (self + other).

        Args:
            other (Tensor | NDArrayLike): Right-hand operand.

        Returns:
            Tensor: Element-wise self + other.
        """
        return _ops_mod.add(self, other)

    def __radd__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise addition with reflected operands (other + self).

        Args:
            other (Tensor | NDArrayLike): Left-hand operand.

        Returns:
            Tensor: Element-wise other + self.
        """
        return _ops_mod.add(other, self)

    def __sub__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise subtraction (self - other).

        Args:
            other (Tensor | NDArrayLike): Right-hand operand.

        Returns:
            Tensor: Element-wise self - other.
        """
        return _ops_mod.subtract(self, other)

    def __rsub__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise subtraction with reflected operands (other - self).

        Args:
            other (Tensor | NDArrayLike): Left-hand operand.

        Returns:
            Tensor: Element-wise other - self.
        """
        return _ops_mod.subtract(other, self)

    def __mul__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise multiplication (self * other).

        Args:
            other (Tensor | NDArrayLike): Right-hand operand.

        Returns:
            Tensor: Element-wise self * other.
        """
        return _ops_mod.multiply(self, other)

    def __rmul__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise multiplication with reflected operands (other * self).

        Args:
            other (Tensor | NDArrayLike): Left-hand operand.

        Returns:
            Tensor: Element-wise other * self.
        """
        return _ops_mod.multiply(other, self)

    def __truediv__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise division (self / other).

        Args:
            other (Tensor | NDArrayLike): Divisor.

        Returns:
            Tensor: Element-wise self / other.
        """
        return _ops_mod.divide(self, other)

    def __rtruediv__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise division with reflected operands (other / self).

        Args:
            other (Tensor | NDArrayLike): Dividend.

        Returns:
            Tensor: Element-wise other / self.
        """
        return _ops_mod.divide(other, self)

    def __pow__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise exponentiation (self ** other).

        Args:
            other (Tensor | NDArrayLike): Exponent.

        Returns:
            Tensor: Element-wise self ** other.
        """
        return _ops_mod.power(self, other)

    def __rpow__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise exponentiation with reflected operands (other ** self).

        Args:
            other (Tensor | NDArrayLike): Base.

        Returns:
            Tensor: Element-wise other ** self.
        """
        return _ops_mod.power(other, self)

    def __matmul__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Matrix multiplication (self @ other).

        Args:
            other (Tensor | NDArrayLike): Right-hand matrix or batch of matrices.

        Returns:
            Tensor: Result of self @ other.
        """
        return _ops_mod.matmul(self, other)

    def __rmatmul__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Matrix multiplication with reflected operands (other @ self).

        Args:
            other (Tensor | NDArrayLike): Left-hand matrix or batch of matrices.

        Returns:
            Tensor: Result of other @ self.
        """
        return _ops_mod.matmul(other, self)

    def __neg__(self) -> Tensor:
        """Element-wise numerical negative (-self).

        Returns:
            Tensor: Element-wise -self.
        """
        return _ops_mod.negative(self)

    def __abs__(self) -> Tensor:
        """Element-wise absolute value (|self|).

        Returns:
            Tensor: Element-wise |self|.
        """
        return _ops_mod.absolute(self)

    # ------------------------------------------------------------------
    # Reduction methods
    # ------------------------------------------------------------------

    def sum(
        self,
        *,
        axis: int | tuple[int, ...] | list[int] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Sum of elements along a given axis.

        Args:
            axis (int | tuple[int, ...] | list[int] | None, optional): Axis or axes to
                reduce. If None, all elements are summed. Defaults to None.
            keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
                Defaults to False.

        Returns:
            Tensor: Sum of elements along the specified axis.
        """
        return _ops_mod.sum(self, axis=axis, keepdims=keepdims)

    def mean(
        self,
        *,
        axis: int | tuple[int, ...] | list[int] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Arithmetic mean along a given axis.

        Args:
            axis (int | tuple[int, ...] | list[int] | None, optional): Axis or axes to
                reduce. If None, the mean of all elements is computed. Defaults to None.
            keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
                Defaults to False.

        Returns:
            Tensor: Mean of elements along the specified axis.
        """
        return _ops_mod.mean(self, axis=axis, keepdims=keepdims)

    def max(
        self,
        *,
        axis: int | tuple[int, ...] | list[int] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Maximum along a given axis.

        Args:
            axis (int | tuple[int, ...] | list[int] | None, optional): Axis or axes to
                reduce. If None, the maximum over all elements is returned.
                Defaults to None.
            keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
                Defaults to False.

        Returns:
            Tensor: Maximum value along the specified axis.
        """
        return _ops_mod.max(self, axis=axis, keepdims=keepdims)

    def min(
        self,
        *,
        axis: int | tuple[int, ...] | list[int] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Minimum along a given axis.

        Args:
            axis (int | tuple[int, ...] | list[int] | None, optional): Axis or axes to
                reduce. If None, the minimum over all elements is returned.
                Defaults to None.
            keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
                Defaults to False.

        Returns:
            Tensor: Minimum value along the specified axis.
        """
        return _ops_mod.min(self, axis=axis, keepdims=keepdims)

    def argmax(
        self,
        *,
        axis: int | tuple[int, ...] | list[int] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Indices of the maximum values along a given axis.

        Args:
            axis (int | tuple[int, ...] | list[int] | None, optional): Axis along which
                to find the maximum. If None, operates on the flattened array.
                Defaults to None.
            keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
                Defaults to False.

        Returns:
            Tensor: Integer indices of the maximum values (requires_grad=False).
        """
        return _ops_mod.argmax(self, axis=axis, keepdims=keepdims)

    def argmin(
        self,
        *,
        axis: int | tuple[int, ...] | list[int] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        """Indices of the minimum values along a given axis.

        Args:
            axis (int | tuple[int, ...] | list[int] | None, optional): Axis along which
                to find the minimum. If None, operates on the flattened array.
                Defaults to None.
            keepdims (bool, optional): Whether to keep reduced dimensions as size-1.
                Defaults to False.

        Returns:
            Tensor: Integer indices of the minimum values (requires_grad=False).
        """
        return _ops_mod.argmin(self, axis=axis, keepdims=keepdims)

    # ------------------------------------------------------------------
    # Unary / element-wise methods
    # ------------------------------------------------------------------

    def sqrt(self) -> Tensor:
        """Element-wise square root.

        Returns:
            Tensor: Element-wise sqrt(x).
        """
        return _ops_mod.sqrt(self)

    def square(self) -> Tensor:
        """Element-wise square.

        Returns:
            Tensor: Element-wise x ** 2.
        """
        return _ops_mod.square(self)

    def exp(self) -> Tensor:
        """Element-wise natural exponential.

        Returns:
            Tensor: Element-wise e ** x.
        """
        return _ops_mod.exp(self)

    def log(self) -> Tensor:
        """Element-wise natural logarithm.

        Returns:
            Tensor: Element-wise ln(x).
        """
        return _ops_mod.log(self)

    def sin(self) -> Tensor:
        """Element-wise sine.

        Returns:
            Tensor: Element-wise sin(x), where x is in radians.
        """
        return _ops_mod.sin(self)

    def cos(self) -> Tensor:
        """Element-wise cosine.

        Returns:
            Tensor: Element-wise cos(x), where x is in radians.
        """
        return _ops_mod.cos(self)

    # ------------------------------------------------------------------
    # Shape methods
    # ------------------------------------------------------------------

    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        """Reshape the tensor to the given shape.

        Args:
            shape (tuple[int, ...]): Target shape. Must be compatible with the total
                number of elements in this Tensor.

        Returns:
            Tensor: Tensor with the same data viewed under the new shape.
        """
        return _ops_mod.reshape(self, shape=shape)

    # ------------------------------------------------------------------
    # Comparison dunder methods (not differentiable, return bool tensors)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> Tensor:  # type: ignore[override]
        """Element-wise equality check (self == other).

        Args:
            other (object): Value to compare against. Must be on the same device
                if it is a Tensor or NDArray.

        Returns:
            Tensor: Boolean tensor, True where elements are equal (requires_grad=False).
        """
        _ensure_comparable(self, other)
        if isinstance(other, Tensor):
            return Tensor(self._xp.asarray(self.data == other.data), requires_grad=False)
        return Tensor(self._xp.asarray(self.data == other), requires_grad=False)

    def __ne__(self, other: object) -> Tensor:  # type: ignore[override]
        """Element-wise inequality check (self != other).

        Args:
            other (object): Value to compare against. Must be on the same device
                if it is a Tensor or NDArray.

        Returns:
            Tensor: Boolean tensor, True where elements differ (requires_grad=False).
        """
        _ensure_comparable(self, other)
        if isinstance(other, Tensor):
            return Tensor(self._xp.asarray(self.data != other.data), requires_grad=False)
        return Tensor(self._xp.asarray(self.data != other), requires_grad=False)

    def __lt__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise less-than check (self < other).

        Args:
            other (Tensor | NDArrayLike): Right-hand operand. Must be on the same device.

        Returns:
            Tensor: Boolean tensor, True where self < other (requires_grad=False).
        """
        _ensure_comparable(self, other)
        if isinstance(other, Tensor):
            return Tensor(self._xp.asarray(self.data < other.data), requires_grad=False)
        return Tensor(self._xp.asarray(self.data < other), requires_grad=False)

    def __le__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise less-than-or-equal check (self <= other).

        Args:
            other (Tensor | NDArrayLike): Right-hand operand. Must be on the same device.

        Returns:
            Tensor: Boolean tensor, True where self <= other (requires_grad=False).
        """
        _ensure_comparable(self, other)
        if isinstance(other, Tensor):
            return Tensor(self._xp.asarray(self.data <= other.data), requires_grad=False)
        return Tensor(self._xp.asarray(self.data <= other), requires_grad=False)

    def __gt__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise greater-than check (self > other).

        Args:
            other (Tensor | NDArrayLike): Right-hand operand. Must be on the same device.

        Returns:
            Tensor: Boolean tensor, True where self > other (requires_grad=False).
        """
        _ensure_comparable(self, other)
        if isinstance(other, Tensor):
            return Tensor(self._xp.asarray(self.data > other.data), requires_grad=False)
        return Tensor(self._xp.asarray(self.data > other), requires_grad=False)

    def __ge__(self, other: Tensor | NDArrayLike) -> Tensor:
        """Element-wise greater-than-or-equal check (self >= other).

        Args:
            other (Tensor | NDArrayLike): Right-hand operand. Must be on the same device.

        Returns:
            Tensor: Boolean tensor, True where self >= other (requires_grad=False).
        """
        _ensure_comparable(self, other)
        if isinstance(other, Tensor):
            return Tensor(self._xp.asarray(self.data >= other.data), requires_grad=False)
        return Tensor(self._xp.asarray(self.data >= other), requires_grad=False)

    # ------------------------------------------------------------------
    # Shape / utility dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Length of the first dimension.

        Returns:
            int: Size of axis 0.
        """
        return self.shape[0]

    def __getitem__(self, key: Any) -> Tensor:
        """Index into the tensor.

        Args:
            key (Any): An index, slice, integer array, or tuple thereof.
                Tensor indices are automatically unwrapped to their underlying arrays.

        Returns:
            Tensor: The selected sub-tensor (requires_grad=False).
        """
        result = self.data[_unwrap_tensor_index(key)]
        return Tensor(self._xp.asarray(result))

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
            str: String of the form ``Tensor(<data>)``.
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
                ``Tensor(<data>, dtype=..., device='...', requires_grad=...)``.
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

        super().__init__(
            data,
            requires_grad=True,
            keep_grad=True,
        )
        if self.dtype.is_integer:
            raise ValueError("Parameter must have float type, found int.")
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
    dtype: TensorDType | None = None,
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
        dtype (TensorDType | None, optional): The data type of the array data.
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
    dtype: TensorDType | None = None,
    device: DeviceLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor filled with ones.

    Args:
        shape (tuple[int, ...]): Shape of the resulting Tensor.
        dtype (TensorDType | None, optional): Data type. If None, will be Float32. Defaults to None.
        device (DeviceLike | None, optional): Target device. If None, will be CPU. Defaults to None.
        requires_grad (bool, optional): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: A Tensor of ones.
    """
    return tensor(
        np.ones(shape),
        dtype=dtype or Float32(),
        device=device,
        requires_grad=requires_grad,
    )


def zeros(
    shape: tuple[int, ...],
    *,
    dtype: TensorDType | None = None,
    device: DeviceLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor filled with zeros.

    Args:
        shape (tuple[int, ...]): Shape of the resulting Tensor.
        dtype (TensorDType | None, optional): Data type. If None, will be Float32. Defaults to None.
        device (DeviceLike | None, optional): Target device. If None, will be CPU. Defaults to None.
        requires_grad (bool, optional): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: A Tensor of zeros.
    """
    return tensor(
        np.zeros(shape),
        dtype=dtype or Float32(),
        device=device,
        requires_grad=requires_grad,
    )


def eye(
    n: int,
    m: int | None = None,
    *,
    dtype: TensorDType | None = None,
    device: DeviceLike | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a 2D Tensor with ones on the diagonal and zeros elsewhere.

    Args:
        n (int): Number of rows.
        m (int | None, optional): Number of columns. If None, uses `n`.
            Defaults to None.
        dtype (TensorDType | None, optional): Data type. If None, will be Float32.
            Defaults to None.
        device (DeviceLike | None, optional): Target device. If None, will be CPU.
            Defaults to None.
        requires_grad (bool, optional): Whether to track gradients. Defaults to False.

    Returns:
        Tensor: An identity-like Tensor.
    """
    return tensor(
        np.eye(n, m),
        dtype=dtype or Float32(),
        device=device,
        requires_grad=requires_grad,
    )


def ones_like(
    other: Tensor,
    *,
    dtype: TensorDType | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor of ones matching the shape and device of `other`.

    Args:
        other (Tensor): Reference tensor for shape and device.
        dtype (TensorDType | None, optional): Override dtype. If None, dtype will
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
    dtype: TensorDType | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a Tensor of zeros matching the shape and device of `other`.

    Args:
        other (Tensor): Reference tensor for shape and device.
        dtype (TensorDType | None, optional): Override dtype. If None, dtype will
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
