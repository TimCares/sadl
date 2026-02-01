"""Neural Network definitions. Each Layer is defined as a mathematical function."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, cast

from .backend import TensorDevice, xp
from .tensor import Parameter, no_grad_fn

if TYPE_CHECKING:
    from collections.abc import Callable, ValuesView

    from . import tensor as sadl


logger = logging.getLogger(__name__)


RNG = xp.random.default_rng()


class Function(ABC):
    """Abstract Base Class (ABC) for all Neural Network related layers.

    Note:
        Parameters and nested Functions must be stored in mutable containers
        (direct attributes, lists, or dicts). Storing them in tuples or sets
        is not allowed because:
        - Tuples are immutable, so parameters cannot be replaced during
          operations like `copy_to_device`.
        - Sets have no stable ordering, making parameter access unpredictable.
    """

    @abstractmethod
    def __call__(self, x: sadl.Tensor, **kwargs: Any) -> Any:
        """Forward pass.

        Args:
            x (sadl.Tensor): Input
            **kwargs (Any): Additional input.

        Returns:
            Any: Some transformed output.
        """

    def traverse_parameters(
        self,
        on_parameter: Callable[[str, sadl.Parameter], sadl.Parameter | None],
    ) -> None:
        """Recursively traverse all Parameters and optionally transform them.

        This is the core traversal logic used by `get_parameters`, `load_parameters`,
        and `copy_to_device`.

        Args:
            on_parameter: Callback called for each Parameter with (path, param).
                If it returns a Parameter, the original is replaced in-place.
                If it returns None, no replacement occurs.
        """

        def handle_item(
            parent: Any,
            key: str | int,
            item: Any,
            path: str,
        ) -> None:
            if isinstance(item, Parameter):
                result = on_parameter(path, item)
                if result is not None:
                    # Replace the parameter in its parent container
                    if isinstance(parent, dict):
                        parent[key] = result
                    elif isinstance(parent, list):
                        parent[key] = result  # type: ignore[index]
                    elif hasattr(parent, "__dict__") and isinstance(key, str):
                        setattr(parent, key, result)
                    # Note: tuples are immutable, can't replace items in them
            elif isinstance(item, Function):
                # Recurse into nested Functions
                item.traverse_parameters(lambda p, param: on_parameter(f"{path}.{p}", param))
            elif isinstance(item, set):
                for elem in item:
                    if isinstance(elem, (Parameter, Function)):
                        raise TypeError(
                            f'Found "{type(elem).__name__}" in property of type "set" '
                            f'for path "{path}". Sets are not allowed for storing '
                            f"Parameters or Functions (no stable ordering)."
                        )
            elif isinstance(item, tuple):
                for i, elem in enumerate(item):
                    if isinstance(elem, (Parameter, Function)):
                        raise TypeError(
                            f'Found "{type(elem).__name__}" in property of type "tuple" '
                            f'for path "{path}[{i}]". Tuples are not allowed for storing '
                            f"Parameters or Functions (immutable, cannot replace during "
                            f"copy_to_device). Use a list instead."
                        )
                    # Still traverse nested structures (e.g., tuple containing a dict)
                    handle_item(item, i, elem, f"{path}[{i}]")
            elif isinstance(item, list):
                for i, elem in enumerate(item):
                    handle_item(item, i, elem, f"{path}[{i}]")
            elif isinstance(item, dict):
                for k, v in item.items():
                    handle_item(item, k, v, f"{path}{{{k}}}")

        for attr_name, attr_value in vars(self).items():
            handle_item(self, attr_name, attr_value, attr_name)

    @property
    def requires_grad(self) -> bool:
        """Defines if **all** parameters of the function require a gradient.

        Returns:
            bool: Returns `True` if all parameters require
                a gradient, meaning the Function itself
                requires a gradient. If just **one**
                Parameter does not require a gradient
                (`param.requires_grad==False`), it will
                return `False`.
        """
        return all(param.requires_grad for param in self.parameters)

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set whether the function requires a gradient.

        This automatically sets `requires_grad` for
        **all** parameters of the Function to `value`.

        Args:
            value (bool): Whether to make the function
                require a grad (`True`), or not (`False`).
        """
        for param in self.parameters:
            param.requires_grad = value

    @property
    def device(self) -> tuple[TensorDevice] | None:
        """The devices on which the Function is currently located.

        Can be mutliple if the function is sharded across multiple
        devices.

        Returns:
            tuple[TensorDevice] | None: The devices on which the function
                parameters are currently located. None, if the function has
                no parameters.
        """
        unique_devices = {p.device for p in self.parameters}
        if len(unique_devices) == 0:
            return None
        return tuple(unique_devices)

    # "no_grad_fn" technically not needed here, as Parameter.copy_to_device
    # is a utility operation (not tracked in the computation graph).
    # We still annotate just to be safe.
    @no_grad_fn
    def copy_to_device(self, device: TensorDevice) -> Function:
        """Copy the function to the specified `device`.

        Copies all its parameters under the hood.

        Args:
            device (TensorDevice): The device to copy the function to.

        Returns:
            Function: self, for method chaining.
        """

        def copy_param(_path: str, param: sadl.Parameter) -> sadl.Parameter:
            return param.copy_to_device(device)

        self.traverse_parameters(copy_param)
        return self

    def is_training(self) -> bool:
        """If the function is in training mode or not.

        Returns:
            bool: `True` of all function parameters
                are in training mode (relevant of e.g. Dropout
                or BatchNorm).
        """
        return all(param.is_training for param in self.parameters)

    def _set_train_state(self, *, is_training: bool) -> None:
        """Small helper."""
        for param in self.parameters:
            param.is_training = is_training

    def train(self) -> Function:
        """Set all function parameters to training mode.

        Returns:
            Function: Self, for chaining.
        """
        self._set_train_state(is_training=True)
        return self

    def inference(self) -> Function:
        """Set all function parameters to inference mode.

        Returns:
            Function: Self, for chaining.
        """
        self._set_train_state(is_training=False)
        return self

    # "no_grad_fn" technically not needed here, as Parameter.copy_to_device
    # is a utility operation (not tracked in the computation graph).
    # We still annotate just to be safe.
    @no_grad_fn
    def get_parameters(
        self,
        to_device: TensorDevice | None = None,
    ) -> OrderedDict[str, sadl.Parameter]:
        """Recursively collect all Parameters from this Function and nested children.

        Args:
            to_device (TensorDevice | None): If specified, copy each
                Parameter to this device in the returned dict. Defaults to None.

        Returns:
            OrderedDict[str, sadl.Parameter]: Parameters keyed by their path,
                e.g. "layers[0].W", "layers[1].b", "payload{key}.W".
        """
        result: OrderedDict[str, sadl.Parameter] = OrderedDict()

        def collect_param(path: str, param: sadl.Parameter) -> None:
            result[path] = param.copy_to_device(to_device) if to_device is not None else param

        self.traverse_parameters(collect_param)
        return result

    @property
    def parameters(self) -> ValuesView[sadl.Parameter]:
        """The parameters of the function.

        Returns:
            ValuesView[sadl.Parameter]: A view over the
                function parameters.
        """
        return self.get_parameters().values()

    # "no_grad_fn" technically not needed here, as Parameter.copy_to_device
    # is a utility operation (not tracked in the computation graph).
    # We still annotate just to be safe.
    @no_grad_fn
    def load_parameters(
        self,
        *,
        parameters: OrderedDict[str, sadl.Parameter],
        match_function_device: bool = False,
        partial: bool = False,
    ) -> Function:
        """Load parameters into this Function from a parameter dict.

        Args:
            parameters (OrderedDict[str, sadl.Parameter]): Parameters keyed by path,
                as returned by `get_parameters`.
            match_function_device (bool): If True, copy each loaded
                parameter to the target parameter's device before assignment.
                If False, raises on device mismatch. Defaults to False.
            partial (bool): If True, allow missing keys in `parameters`.
                If False, raises on missing keys. Defaults to False.

        Returns:
            Function: self, for method chaining.
        """

        def load_param(path: str, param: sadl.Parameter) -> None:
            init_data = parameters.get(path)
            if init_data is None:
                if not partial:
                    raise KeyError(f'Parameter data not found for "{path}"')
                return  # Skip this parameter

            if param.shape != init_data.shape:
                raise ValueError(
                    f'Shape mismatch for parameter "{path}". '
                    f'Found "{param.shape}", expected "{init_data.shape}".'
                )

            if match_function_device:
                init_data = init_data.copy_to_device(param.device)
            elif param.device != init_data.device:
                raise ValueError(
                    f'Device mismatch for parameter "{path}". '
                    f'Found "{init_data.device}", expected "{param.device}".'
                )

            param[...] = init_data  # in-place assignment
            # we do not return "init_data" here because we only want to
            # modify the parameter **data** (the buffer), not the full object
            return  # Don't replace

        self.traverse_parameters(load_param)
        return self


class Sigmoid(Function):
    """Sigmoid activation function."""

    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:  # type: ignore[override]
        """Forward pass, computes the sigmoid activation function.

        Args:
            x (sadl.Tensor): Input

        Returns:
            sadl.Tensor: Transformed output
        """
        return cast("sadl.Tensor", 1 / (xp.exp(-x) + 1))


class ReLU(Function):
    """ReLU activation function."""

    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:  # type: ignore[override]
        """Forward pass, computes the ReLU activation function.

        Args:
            x (sadl.Tensor): Input

        Returns:
            sadl.Tensor: Transformed output
        """
        return cast("sadl.Tensor", xp.maximum(xp.array([0]), x))


class Linear(Function):
    """Base dense/linear neural network layer.

    Args:
        dim_in (int): Input dimension size.
        dim_out (int): Output dimension size.
        bias (bool): Whether to use a bias. Defaults to True
    """

    def __init__(
        self,
        *,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        dtype: xp.dtype = xp.float32,
    ) -> None:
        self.dim_in = dim_in
        self.dim_out = dim_out
        # Xavier initialization for weights
        self.W = Parameter(
            RNG.random((self.dim_in, self.dim_out)).astype(dtype)
            * xp.sqrt(2.0 / (self.dim_in + self.dim_out))
        )
        self.b = Parameter(xp.zeros((self.dim_out,), dtype=dtype)) if bias else None
        self.INPUT_N_DIM = 2

    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:  # type: ignore[override]
        """Forward pass.

        Args:
            x (sadl.Tensor): Input

        Returns:
            sadl.Tensor: Transformed output
        """
        assert x.ndim == self.INPUT_N_DIM, (
            "Input must have two dimensions, dim[0] -> sample dim, dim[1] -> feature dim"
        )
        assert x.shape[1] == self.dim_in, (
            "Input feature dim must match layer input dim/embedding size"
        )
        x = xp.matmul(x, self.W)
        x = x + self.b if self.b is not None else x
        return cast("sadl.Tensor", x)


class Mlp(Function):
    """Mlp (Base Neural Network).

    Args:
        *layers (Function): A list of Functions.
    """

    def __init__(self, layers: list[Function]) -> None:
        self.layers = layers

    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:  # type: ignore[override]
        """Forward pass.

        Calls all layers subsequently in the order
        as provided in the constructor.

        Args:
            x (sadl.Tensor): Input

        Returns:
            sadl.Tensor: Transformed output
        """
        for layer in self.layers:
            x = layer(x)
        return x


__all__ = [
    "Function",
    "Linear",
    "Mlp",
    "ReLU",
    "Sigmoid",
]
