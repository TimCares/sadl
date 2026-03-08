"""DType wrapper implementation around the backend (cupy and numpy) dtypes."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, TypeAlias

import numpy as np

from .array_provider import NDArrayLike, xp

BackendDType: TypeAlias = np.dtype[Any] | type[np.generic]


class TensorDType(ABC):  # noqa: B024
    """Wrapper class around the backend (cupy and numpy) dtypes.

    Each concrete subclass is automatically registered by its lowercased class
    name (e.g. `Float32` -> `"float32"`) via `__init_subclass__`, which is
    the key used by `from_backend` to look up the matching wrapper.
    """

    _registry: ClassVar[dict[str, type[TensorDType]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        TensorDType._registry[cls.__name__.lower()] = cls

    def to_backend(self) -> BackendDType:
        """Convert the sadl dtype wrapper to the backend dtype.

        Returns:
            BackendDType: The cupy or numpy dtype.
        """
        return getattr(xp, self.__class__.__name__.lower())

    @classmethod
    def from_backend(cls, backend_dtype: BackendDType) -> TensorDType:
        """Creates a TensorDType from a backend dtype.

        Accepts any cupy or numpy dtype (e.g. `xp.float32`, `np.int8`).
        The canonical name is resolved via `np.dtype` so both dtype *types*
        and dtype *instances* are accepted.

        Args:
            backend_dtype (BackendDType): The cupy or numpy dtype.

        Raises:
            ValueError: If the backend dtype has no corresponding sadl wrapper.

        Returns:
            TensorDType: The matching sadl dtype wrapper.
        """
        name = np.dtype(backend_dtype).name
        dtype_cls = TensorDType._registry.get(name)
        if dtype_cls is None:
            raise ValueError(f"Dtype '{name}' is not supported in sadl.")
        return dtype_cls()

    @classmethod
    def from_ndarraylike(cls, ndarraylike: NDArrayLike) -> TensorDType:
        """Infer the sadl dtype from any NDArrayLike value.

        Arrays carry an explicit dtype; scalars and lists fall back to the
        corresponding default dtype.

        Args:
            ndarraylike (NDArrayLike): An ndarray, scalar, or flat list of scalars.

        Raises:
            ValueError: If the dtype cannot be determined or is unsupported.

        Returns:
            TensorDType: The inferred sadl dtype.
        """
        if hasattr(ndarraylike, "dtype"):
            return cls.from_backend(getattr(ndarraylike, "dtype", np.float32))

        if isinstance(ndarraylike, int):
            return DEFAULT_INT_DTYPE()

        if isinstance(ndarraylike, float):
            return DEFAULT_FLOAT_DTYPE()

        if isinstance(ndarraylike, list):
            elem_types: set[type] = {type(elem) for elem in ndarraylike}
            if elem_types <= {int}:
                return DEFAULT_INT_DTYPE()
            if elem_types <= {float}:
                return DEFAULT_FLOAT_DTYPE()
            if elem_types <= {int, float}:
                return DEFAULT_FLOAT_DTYPE()
            if elem_types <= {list}:
                # Nested list (e.g. 2-D): recurse on the first sub-list to
                # determine the scalar dtype.
                return cls.from_ndarraylike(ndarraylike[0])
            raise ValueError(f"List elements must be int, float, or list, got: {elem_types!r}")

        raise ValueError(f"Cannot infer dtype from type {type(ndarraylike)!r}.")

    @property
    def is_floating_point(self) -> bool:
        """True for all floating-point dtypes (float16/32/64/96/128)."""
        return np.issubdtype(self.to_backend(), np.floating)

    @property
    def is_integer(self) -> bool:
        """True for all integer dtypes, both signed and unsigned."""
        return np.issubdtype(self.to_backend(), np.integer)

    @property
    def is_signed(self) -> bool:
        """True for signed integer and all floating-point dtypes."""
        return np.issubdtype(self.to_backend(), np.signedinteger) or self.is_floating_point

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TensorDType):
            return type(self) is type(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        return f"sadl.{self.__class__.__name__.lower()}"


class Int8(TensorDType):
    """8-bit signed integer."""


class Int16(TensorDType):
    """16-bit signed integer."""


class Int32(TensorDType):
    """32-bit signed integer."""


class Int64(TensorDType):
    """64-bit signed integer."""


class UInt8(TensorDType):
    """8-bit unsigned integer."""


class UInt16(TensorDType):
    """16-bit unsigned integer."""


class UInt32(TensorDType):
    """32-bit unsigned integer."""


class UInt64(TensorDType):
    """64-bit unsigned integer."""


class Bool(TensorDType):
    """Boolean dtype."""


class Float16(TensorDType):
    """16-bit (half-precision) floating point."""


class Float32(TensorDType):
    """32-bit (single-precision) floating point."""


class Float64(TensorDType):
    """64-bit (double-precision) floating point."""


class Float96(TensorDType):
    """96-bit extended-precision floating point (platform-dependent)."""


class Float128(TensorDType):
    """128-bit (quad-precision) floating point (platform-dependent)."""


DEFAULT_INT_DTYPE: TypeAlias = Int32
DEFAULT_FLOAT_DTYPE: TypeAlias = Float32
DEFAULT_DTYPE: TypeAlias = Float32


__all__ = [
    "DEFAULT_DTYPE",
    "DEFAULT_FLOAT_DTYPE",
    "DEFAULT_INT_DTYPE",
    "BackendDType",
    "Bool",
    "Float16",
    "Float32",
    "Float64",
    "Float96",
    "Float128",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "TensorDType",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
]
