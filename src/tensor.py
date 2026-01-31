"""Custom tensor implementations that support autograd."""

from __future__ import annotations

import logging
import struct
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ParamSpec, Self, TypeVar

import numpy as np

from .backend import BACKEND, TensorDevice, xp
from .grad_ops import GradOp, get_grad_op, normalize_grad_op_name
from .utils import copy_array

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from types import TracebackType

P = ParamSpec("P")
T = TypeVar("T")


logger = logging.getLogger(__name__)


def _to_array(x: Any) -> Any:
    """Recursively convert Tensors to plain ndarrays (handles nested lists/tuples)."""
    if isinstance(x, Tensor):
        return xp.asarray(x)
    if isinstance(x, list | tuple):
        converted = [_to_array(i) for i in x]
        return type(x)(converted)
    return x


def _to_tensor(x: Any) -> Tensor:
    """Convert input to Tensor. Non-Tensors become Tensors with requires_grad=False."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=False)


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


class Tensor(xp.ndarray):  # type: ignore[misc]
    """A tensor wrapper around arrays with autograd support."""

    def __init__(  # noqa: PLR0913
        self,
        data: Any = None,  # noqa: ARG002 - Ignored, handled by __new__, needed for signature
        *,
        src: tuple[Tensor, ...] | None = None,
        creator_op: str | None = None,
        op_ctx: dict[str, Any] | None = None,
        requires_grad: bool = False,
        keep_grad: bool = False,
    ) -> None:
        self.src: tuple[Tensor, ...] = src or ()

        backward_fn = get_grad_op(creator_op) if creator_op else None

        if not self.is_leaf() and backward_fn is None:
            raise ValueError(f'Gradient propagation not supported for op "{creator_op}"')

        self.backward_fn: GradOp | None = backward_fn
        self.op_ctx: dict[str, Any] = op_ctx or {}

        self.requires_grad = _GRAD_MODE_ENABLED and requires_grad

        self.grad: xp.array | None = None

        self.keep_grad = keep_grad

    def __array_finalize__(self, obj: Any) -> None:
        """Called when a new Tensor is created via .view(), slicing, or ufuncs.

        Sets default values for all Tensor attributes. These can be overridden
        after creation if needed.
        """
        if obj is None:
            # Called from __new__ via explicit constructor - __init__ will handle it
            return
        # Copy attributes from source object if available, otherwise use defaults
        # These assignments are intentionally duplicated from __init__ because
        # __array_finalize__ is called for views/slices instead of __init__
        self.src: tuple[Tensor, ...] = getattr(obj, "src", ())  # type: ignore[no-redef]
        self.backward_fn: GradOp | None = getattr(obj, "backward_fn", None)  # type: ignore[no-redef]
        self.op_ctx: dict[str, Any] = getattr(obj, "op_ctx", {})  # type: ignore[no-redef]
        self.requires_grad: bool = getattr(obj, "requires_grad", False)  # type: ignore[no-redef]
        self.grad: xp.array | None = getattr(obj, "grad", None)  # type: ignore[no-redef]
        self.keep_grad: bool = getattr(obj, "keep_grad", False)  # type: ignore[no-redef]

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
        new_device_array = copy_array(array=self, device=device)

        # Check if this is a non-leaf tensor in an active computation graph
        # (like intermediate activations in multi-GPU scenarios)
        src_requires_grad = [s.requires_grad for s in self.src]
        is_in_graph = (
            _GRAD_MODE_ENABLED
            and self.requires_grad
            and len(self.src) > 0
            and any(src_requires_grad)
        )

        # __array_finalize__ sets defaults from new_device_array (plain xp.ndarray),
        # so we manually set attributes
        result: Tensor = new_device_array.view(Tensor)

        if is_in_graph:
            # Tracked operation: gradients flow back through device transfer
            result.src = (self,)
            result.backward_fn = get_grad_op("copy_to_device")
            result.requires_grad = True
        else:
            # Utility operation for leaf tensors
            result.requires_grad = self.requires_grad

        result.keep_grad = self.keep_grad
        return result

    def detach(
        self,
        *,
        in_place: bool = False,
    ) -> Tensor:
        """Detatch the Tensor from the computation graph.

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
            self.src = ()
            self.backward_fn = None
            self.op_ctx = {}
            if not self.keep_grad:
                self.grad = None
            return self
        detached_tensor = Tensor(
            self.copy(),
            requires_grad=self.requires_grad,
            keep_grad=self.keep_grad,
        )
        detached_tensor.grad = self.grad if self.keep_grad else None
        return detached_tensor

    def cpu(self) -> Tensor:
        """Move the Tensor to the cpu.

        Note: If the Tensor already is on the cpu,
        no copy is created. Instead, the Tensor is returned as is.

        Returns:
            Tensor: A **copy** of the Tensor on the cpu,
                if it wasn't on the cpu before.
        """
        return self.copy_to_device(device="cpu")

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
        return self.copy_to_device(device=device_id)

    def __hash__(self) -> int:
        """Identity-based hash for use in sets/dicts (computation graph tracking)."""
        return id(self)

    def __new__(cls, data: Iterable[Any], **kwargs: Any) -> Self:
        """Initializes the data."""
        # **kwargs accepts src, creator_op, etc. but we don't use them here
        # They'll be handled by __init__
        result: Self = xp.asarray(data, dtype=kwargs.get("dtype")).view(cls)
        return result

    def __array_ufunc__(
        self,
        ufunc: xp.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        logger.debug(
            '__array_ufunc__: ufunc="%s" method="%s" inputs="%s" kwargs="%s"',
            ufunc.__name__,
            method,
            inputs,
            kwargs,
        )

        xp_input_arrays = tuple(_to_array(x) for x in inputs)
        kwargs = {k: _to_array(v) for k, v in kwargs.items()}

        track_kwargs = kwargs.copy()

        func = getattr(ufunc, method)

        if func.__name__ in ["maximum", "minimum"]:
            result = func(*xp_input_arrays, **kwargs)
            # create a mask for each axis, where True means the value in x
            #   at this position is an extremum (maximum or minimum, depending on "method"):
            x_mask = result == xp_input_arrays[0]
            # just for completness we do "* kwargs.get("where", 1)" in the following line
            #   this is because if "where" is "False" at a location, the extremum operation
            #       should not be used, and therefore x_mask does not apply
            #   (this is not strictly neccessary, because we account for this in the backward
            #   function by setting the grad to "0" for both x and y at these locations anyway)
            track_kwargs["x_mask"] = x_mask * kwargs.get("where", 1)

        else:
            result = func(*xp_input_arrays, **kwargs)

        # Skip graph building when grad mode is disabled (e.g., during backward pass)
        if not _GRAD_MODE_ENABLED:
            return Tensor(result, requires_grad=False)

        src = tuple(_to_tensor(i) for i in inputs)
        creator_op = normalize_grad_op_name(name=ufunc.__name__, is_reduce=method == "reduce")

        return Tensor(
            result,
            src=src,
            creator_op=creator_op,
            op_ctx=track_kwargs,
            requires_grad=any(elem.requires_grad for elem in src),
            **kwargs,
        )

    def __array_function__(
        self,
        func: Any,
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        logger.debug(
            '__array_function__: func="%s" types="%s" args="%s" kwargs="%s"',
            func.__name__,
            types,
            args,
            kwargs,
        )

        xp_input_arrays = tuple(_to_array(x) for x in args)
        kwargs = {k: _to_array(v) for k, v in kwargs.items()}

        track_kwargs = kwargs.copy()

        if func.__name__ in ["max", "min"]:
            # execute the function, but retain the dimensions:
            keepdims_kwargs = kwargs.copy()
            keepdims_kwargs["keepdims"] = True
            result = func(*xp_input_arrays, **keepdims_kwargs)
            # create a mask for each axis, where True means the value in x
            #   at this position is an extremum (maximum or minimum, depending on "method"):
            x_mask = result == xp_input_arrays[0]
            track_kwargs["x_mask"] = x_mask

            result = (
                result
                if kwargs.get("keepdims", False)
                else xp.squeeze(result, axis=kwargs.get("axis"))
            )
        else:
            result = func(*xp_input_arrays, **kwargs)

        # Skip graph building when grad mode is disabled (e.g., during backward pass)
        if not _GRAD_MODE_ENABLED:
            return Tensor(result, requires_grad=False)

        src = tuple(_to_tensor(a) for a in args)

        return Tensor(
            result,
            src=src,
            creator_op=func.__name__,
            op_ctx=track_kwargs,
            requires_grad=any(elem.requires_grad for elem in src),
            **kwargs,
        )


class Parameter(Tensor):
    """A special Tensor that should be part of a model to optimize.

    Parameters have an additional `is_training` attribute for controlling
    behavior of layers like Dropout and BatchNorm.
    """

    def __init__(
        self,
        data: Any = None,  # Ignored - handled by __new__, but needed for signature compatibility
        *,
        is_training: bool = True,
    ) -> None:
        super().__init__(
            data=data,
            src=None,
            creator_op=None,
            op_ctx=None,
            requires_grad=True,
            keep_grad=True,
        )
        self.is_training = is_training

    def __array_finalize__(self, obj: Any) -> None:
        """Called when a new Parameter is created via .view(), slicing, or ufuncs.

        Extends Tensor's __array_finalize__ to also handle is_training.
        """
        # First call parent's __array_finalize__ to set Tensor attributes
        super().__array_finalize__(obj)
        # Then set Parameter-specific attributes
        self.is_training: bool = getattr(obj, "is_training", True)  # type: ignore[no-redef]

    def __new__(cls, data: Iterable[Any], **kwargs: Any) -> Self:
        """Initializes the data."""
        # parameters must always be float and can never be int
        #   -> their optimization will unlikely yield pure integer
        #       parameters
        #   -> integer parameters are not differentiable!
        #       this is because there exist no infinitesimal steps,
        #       there is always a jump, e.g. 1 -> 2
        dtype = kwargs.get("dtype")
        if xp.issubdtype(dtype, xp.integer):
            raise ValueError("Parameter must have float type, found int.")
        return Tensor.__new__(
            cls=cls,
            data=data,
            **kwargs,
        )

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
        new_device_array = copy_array(array=self, device=device)

        # __array_finalize__ sets defaults, so we manually copy from self
        result: Parameter = new_device_array.view(Parameter)
        result.requires_grad = self.requires_grad
        result.keep_grad = self.keep_grad
        result.is_training = self.is_training
        return result


def tensor(
    data: Any,
    *,
    dtype: Any = None,
    device: TensorDevice = "cpu",
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Factory function to create a Tensor on the specified device.

    Args:
        data (Any): The array data (can be scalar, list, array, etc).
        dtype (Any): The data type of the array data.
            Defaults to None, meaning dtype is inferred from data.
        device (TensorDevice): The device on which the Tensor
            should be created. Defaults to "cpu".
        requires_grad (bool): Whether to track gradients. Defaults to False.
        keep_grad (bool): Whether to retain gradients after backward. Defaults to False.

    Returns:
        Tensor: The created Tensor.
    """
    if BACKEND == "numpy" or device == "cpu":
        arr = xp.array(data, dtype=dtype)
    else:
        with xp.cuda.Device(device):
            arr = xp.array(data, dtype=dtype)

    result: Tensor = arr.view(Tensor)
    # __array_finalize__ sets defaults; override with user values
    result.requires_grad = _GRAD_MODE_ENABLED and requires_grad
    result.keep_grad = keep_grad
    return result


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


# ============================================================================
# Custom binary serialization format (.sadl)
# ============================================================================
#
# Format specification:
# ┌──────────────────────────────────────────────────────────────────────────┐
# │ HEADER                                                                   │
# │   Magic bytes: "TIM\x00"                              (4 bytes)          │
# │   Version: 1                                          (1 byte, uint8)    │
# │   Num tensors: N                                      (4 bytes, uint32)  │
# ├──────────────────────────────────────────────────────────────────────────┤
# │ TENSOR ENTRIES (repeated N times)                                        │
# │   Key length                                          (4 bytes, uint32)  │
# │   Key (UTF-8 encoded)                                 (key_length bytes) │
# │   Dtype string length                                 (1 byte, uint8)    │
# │   Dtype string (e.g. "float32")                       (dtype_len bytes)  │
# │   Ndim                                                (1 byte, uint8)    │
# │   Shape (ndim x uint64)                               (8 * ndim bytes)   │
# │   Data (raw bytes, C-contiguous)                      (variable)         │
# └──────────────────────────────────────────────────────────────────────────┘


_SADL_MAGIC = b"SADL"
_SADL_VERSION = 1


def _dtype_to_str(dtype: Any) -> str:
    """Convert numpy/cupy dtype to string representation."""
    return str(np.dtype(dtype).name)


def _str_to_dtype(dtype_str: str) -> Any:
    """Convert string back to numpy dtype."""
    return np.dtype(dtype_str)


def save(data: Tensor | OrderedDict[str, Tensor], file_path: str) -> None:
    """Save Tensor data to disk using custom binary format.

    Args:
        data (Tensor | OrderedDict[str, Tensor]): The data to save,
            can either be a single Tensor or an OrderedDict with strings
            as keys and Tensors as values.
        file_path (str): The file path to which to store the data. Must
            end with ".sadl".

    Raises:
        ValueError: If an OrderedDict with non-Tensor values is passed.
        ValueError: If file_path doesn't end with ".sadl".
    """
    if not file_path.endswith(".sadl"):
        raise ValueError('file_path must end with ".sadl"')

    # Normalize to OrderedDict
    if isinstance(data, Tensor):
        tensors = OrderedDict([("__single__", data)])
    else:
        if not all(isinstance(v, Tensor) for v in data.values()):
            raise ValueError("If an OrderedDict is passed, all values must be Tensors.")
        tensors = data

    with open(file_path, "wb") as f:
        # Write header
        f.write(_SADL_MAGIC)
        f.write(struct.pack("<B", _SADL_VERSION))  # uint8 version
        f.write(struct.pack("<I", len(tensors)))  # uint32 num tensors

        # Write each tensor
        for key, tensor in tensors.items():
            # Convert to numpy (CPU) for serialization
            arr = np.asarray(tensor)
            # Ensure C-contiguous
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)

            # Key
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<I", len(key_bytes)))  # uint32 key length
            f.write(key_bytes)

            # Dtype
            dtype_str = _dtype_to_str(arr.dtype)
            dtype_bytes = dtype_str.encode("utf-8")
            f.write(struct.pack("<B", len(dtype_bytes)))  # uint8 dtype length
            f.write(dtype_bytes)

            # Shape
            f.write(struct.pack("<B", arr.ndim))  # uint8 ndim
            f.writelines(struct.pack("<Q", dim) for dim in arr.shape)  # uint64 per dimension

            # Raw data
            f.write(arr.tobytes())


def load(file_path: str) -> Tensor | OrderedDict[str, Tensor]:
    """Load Tensor data from disk.

    Args:
        file_path (str): The file path from which to read the data. Must
            end with ".sadl".

    Raises:
        ValueError: If file_path doesn't end with ".sadl".
        ValueError: If file has invalid magic bytes or unsupported version.

    Returns:
        Tensor | OrderedDict[str, Tensor]: The loaded data. Returns a single
            Tensor if one was saved, otherwise an OrderedDict.
    """
    if not file_path.endswith(".sadl"):
        raise ValueError('file_path must end with ".sadl"')

    with open(file_path, "rb") as f:
        # Read and validate header
        magic = f.read(4)
        if magic != _SADL_MAGIC:
            raise ValueError(f"Invalid file format. Expected SADL magic bytes, got {magic!r}")

        version = struct.unpack("<B", f.read(1))[0]
        if version != _SADL_VERSION:
            raise ValueError(f"Unsupported version {version}. Expected {_SADL_VERSION}")

        num_tensors = struct.unpack("<I", f.read(4))[0]

        # Read tensors
        tensors: OrderedDict[str, Tensor] = OrderedDict()
        for _ in range(num_tensors):
            # Key
            key_length = struct.unpack("<I", f.read(4))[0]
            key = f.read(key_length).decode("utf-8")

            # Dtype
            dtype_length = struct.unpack("<B", f.read(1))[0]
            dtype_str = f.read(dtype_length).decode("utf-8")
            dtype = _str_to_dtype(dtype_str)

            # Shape
            ndim = struct.unpack("<B", f.read(1))[0]
            shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(ndim))

            # Data
            num_bytes = int(np.prod(shape)) * dtype.itemsize
            data_bytes = f.read(num_bytes)
            arr = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

            tensors[key] = tensor(arr)

        # Return single tensor if that's what was saved
        if len(tensors) == 1 and "__single__" in tensors:
            return tensors["__single__"]
        return tensors


__all__ = [
    "Parameter",
    "Tensor",
    "load",
    "no_grad",
    "no_grad_fn",
    "ones_like",
    "save",
    "tensor",
    "zeros_like",
]
