import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterable, ValuesView
from itertools import chain

from sadl.ops import zeros_like

from .backend import TensorDevice, xp
from .tensor import Parameter, Tensor, no_grad, no_grad_fn, tensor
from .utils import traverse_attrs

logger = logging.getLogger(__name__)


def toposort(root: Tensor) -> list[Tensor]:
    """Performs topological sort on a graph.

    `root` is the starting point of the graph.

    Args:
        root (Tensor): The starting point of the graph.
            Expected to have an attribute `src` denoting
            a list of its neighbors, also being of type `Tensor`.

    Raises:
        ValueError: If the graph, with the starting point
            given by `root`, is not a DAG.

    Returns:
        list[Tensor]: The ordered nodes.
    """
    ordered_nodes: list[Tensor] = []
    currently_visiting: set[Tensor] = set()
    done: set[Tensor] = set()

    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if node in done:
            continue

        if visited:
            ordered_nodes.append(node)
            currently_visiting.discard(node)
            done.add(node)
            continue

        stack.append((node, True))
        currently_visiting.add(node)
        for neighbor in reversed(node.src):
            if neighbor in done:
                continue
            if neighbor in currently_visiting:
                raise ValueError("Cycle in computation graph detected, but only DAG allowed!")
            stack.append((neighbor, False))

    return ordered_nodes


class Optimizer(ABC):
    """Abstract base class for all optimizers.

    **All optimizer states must be of type `Tensor` or a collection of type `Tensor`**.
    """

    def __init__(self, params: list[Parameter], *, lr: float = 1e-3):
        if len(params) == 0:
            raise ValueError("Must pass at least one parameter to optimize.")
        for param in params:
            if not isinstance(param, Parameter):
                raise TypeError("All parameters passed to the optimizer must be of type Parameter.")
            if not param.keep_grad:
                raise ValueError(
                    "Attribute keep_grad must always be True for all parameters "
                    "to avoid clearing gradients during the backward pass. "
                    "This is important for cases like gradient accumulation."
                )
            if len(param.src) > 0:
                raise ValueError(
                    "Parameters should always be leafes and therefore "
                    "should not have any parents/src "
                    "from which they were created."
                )

        self.params = params
        self.lr = tensor(lr)

    def traverse_state(
        self,
        on_tensor: Callable[[str, Tensor], Tensor | None],
    ) -> None:
        """Recursively traverse all Tensors and optionally transform them.

        Args:
            on_tensor: Callback called for each Tensor with (path, param).
                If it returns a Tensor, the original is replaced in-place.
                If it returns None, no replacement occurs.
        """

        def on_tensor_wrapper(path: str, tensor: Tensor) -> Tensor | None:
            if path.startswith("params"):
                # exlude param to optimize, see self.get_state for more
                return None
            return on_tensor(path, tensor)

        traverse_attrs(
            self,
            target_type=Tensor,
            on_target=on_tensor_wrapper,
        )

    @property
    def device(self) -> tuple[TensorDevice, ...]:
        """The devices on which the optimizer state is currently located.

        Can be mutliple if the optimizer state is sharded across multiple
        devices.

        Returns:
            tuple[TensorDevice, ...]: The devices on which the optimizer
                state is currently located. Empty, if the optimizer
                has no parameters.
        """
        return tuple({attr.device for attr in self.state})

    # "no_grad_fn" technically not needed here, as optimizer state Tensors
    # are leaves (not part of a computation graph). We still annotate
    # just to be safe.
    @no_grad_fn
    def copy_to_device(self, device: TensorDevice) -> "Optimizer":
        """Copy the optimizer state to the specified `device`.

        Args:
            device (TensorDevice): The device to copy the state to.

        Returns:
            Optimizer: self, for method chaining.
        """

        def copy_tensor(_path: str, tensor: Tensor) -> Tensor:
            return tensor.copy_to_device(device)

        self.traverse_state(copy_tensor)
        return self

    # "no_grad_fn" technically not needed here, as optimizer state Tensors
    # are leaves (not part of a computation graph). We still annotate
    # just to be safe.
    @no_grad_fn
    def get_state(self, to_device: TensorDevice | None = None) -> OrderedDict[str, Tensor]:
        """The state of the optimizer.

        Note: Only **direct** attributes of the Optimizer class, which must be of type
        **Tensor** or a collection of type **Tensor** will be considered in the state.
        The `params` attribute, which stores references to the parameters to optimize
        are **not** part of the optimizer state, and will be ignored!

        Args:
            to_device (TensorDevice | None): If specified, copy each
                Tensor in the state to this device in the returned dict.
                If `None`, the device of the Tensors is not changed. Defaults to None.

        Returns:
            OrderedDict[str, Tensor]: Dict containing the state.
        """
        result: OrderedDict[str, Tensor] = OrderedDict()

        def collect_tensors(path: str, tensor: Tensor) -> None:
            result[path] = tensor.copy_to_device(to_device) if to_device is not None else tensor

        self.traverse_state(collect_tensors)
        return result

    @property
    def state(self) -> ValuesView[Tensor]:
        """The Tensors forming the state of the optimizer.

        Returns:
            ValuesView[Tensor]: A view over the
                state Tensors.
        """
        return self.get_state().values()

    # "no_grad_fn" technically not needed here, as optimizer state Tensors
    # are leaves (not part of a computation graph). We still annotate
    # just to be safe.
    @no_grad_fn
    def load_state(
        self,
        *,
        state: OrderedDict[str, Tensor],
        match_device: bool = False,
        partial: bool = False,
    ) -> "Optimizer":
        """Load/initialize the state of the optimizer.

        Args:
            state (OrderedDict[str, Tensor]): The state of the optimizer.
            match_device (bool): If True, copy each loaded Tensor
                to the target's device before assignment.
                If False, raises on device mismatch. Defaults to False.
            partial (bool): If True, allow missing keys in `state`.
                If False, raises on missing keys. Defaults to False.

        Returns:
            Optimizer: self, for method chaining.
        """

        def load_tensor(key: str, tensor: Tensor) -> None:
            init_data = state.get(key, None)
            if init_data is None:
                if not partial:
                    raise KeyError(f'Optimizer state "{key}" not found in passed state!')
                return

            if not isinstance(init_data, Tensor):
                raise TypeError(
                    'Data in passed state must be of type "Tensor", '
                    f'found "{type(init_data).__name__}" ({init_data})'
                )

            if tensor.shape != init_data.shape:
                raise ValueError(
                    f"Shape of seed Tensor does not align with shape of "
                    f'target parameter "{key}". Found "{tensor.shape}", '
                    f'expected "{init_data.shape}".'
                )

            if match_device:
                init_data = init_data.copy_to_device(tensor.device)
            elif tensor.device != init_data.device:
                raise ValueError(
                    f"Device of seed Tensor does not align with device of "
                    f'target parameter "{key}". Found "{init_data.device}", '
                    f'expected "{tensor.device}".'
                )

            tensor[...] = init_data  # in-place assignment
            # we do not return "init_data" here because we only want to
            # modify the tensor **data** (the buffer), not the full object
            return  # Don't replace

        self.traverse_state(load_tensor)
        return self

    def _clear_graph(self, topo_nodes: Iterable[Tensor]) -> None:
        """Clears computation graph structure and gradients after backward pass.

        Removes references to parent tensors, backward functions, and operation
        context to free memory. Gradients are cleared for non-parameter tensors
        (keep_grad=False).

        Args:
            topo_nodes (Iterator[Tensor]): Nodes in topological order from the
                computation graph to clear.
        """
        for node in topo_nodes:
            node.detach(in_place=True)

    def _clear_activation_gradients(self, topo_nodes: Iterable[Tensor]) -> None:
        """Clears gradients of activation tensors before backward pass.

        Ensures activations start with no gradients, preventing stale gradients
        from previous backward passes if tensors are reused. Parameters
        (keep_grad=True) retain their gradients for accumulation.

        Args:
            topo_nodes (Iterator[Tensor]): Nodes in topological order from the
                computation graph whose gradients should be cleared.
        """
        for node in topo_nodes:
            if not node.keep_grad:
                node.grad = None

    def backward(self, loss: Tensor) -> None:
        """Perform backpropagation on the computation graph with respect to `loss`.

        Note: Does **not** support accumulating multiple losses
        into the gradient of `loss`. This only works for parameters.
        Use multiple losses and sum them instead.

        Args:
            loss (Tensor): The loss on with respect to which we perform
                gradient calculations.

        Raises:
            ValueError: If the loss is not a scalar.
        """
        if not isinstance(loss, Tensor) or loss.size > 1:
            raise ValueError("Expected 'loss' argument to be to be a scalar Tensor.")
        if loss.keep_grad:
            raise ValueError(
                "keep_grad=True not supported for the loss Tensor. "
                "Use multiple losses and accumulate/sum them into a new "
                "loss Tensor instead."
            )

        node_order = toposort(loss)

        # Clear the gradients of all activations:
        # (this is necessary to make "node.grad is None" below well-defined)
        self._clear_activation_gradients(topo_nodes=node_order)

        # do not use xp.ones_like(loss) here, because "loss" is a Tensor that requires
        # a gradient, which will trigger "__array_function__" and try to track this
        # -> another fix would be moving it down in the "no_grad()" context
        loss.grad = xp.ones(loss.shape, dtype=loss.dtype)  # seed gradient for loss

        with no_grad():
            for node in reversed(node_order):
                # "node.grad is None" -> Means: In the current computation graph,
                #   this node has not received any gradients, meaning it does not
                #   contribute to the loss => We can skip it
                # "node.is_leaf()" -> Means: In the current computation graph,
                #   this node has no parents to pass gradients to, meaning we can skip it
                # "not any(src_requires_grad)" -> Means: In the current computation graph,
                #   not parent requires a gradient to be passed to it, meaning we can skip
                #   the current node
                src_requires_grad = [t.requires_grad for t in node.src]
                if node.grad is None or node.is_leaf() or not any(src_requires_grad):
                    continue

                if node.backward_fn is None:
                    raise ValueError(
                        f'"backward_fn" for node "{id(node)}" in computation graph is "None"!'
                    )

                logger.debug(f'Calling backward function: "{node.backward_fn.__name__}"')

                src_grads = node.backward_fn(
                    *node.src,
                    compute_grad=src_requires_grad,
                    grad_out=node.grad,
                    **node.op_ctx,
                )
                assert len(src_grads) == len(node.src)

                for src, src_grad in zip(node.src, src_grads, strict=True):
                    if src_grad is None:
                        continue

                    assert src.shape == src_grad.shape
                    current_src_grad = src.grad if src.grad is not None else xp.zeros_like(src)
                    src.grad = current_src_grad + src_grad

        self._clear_graph(topo_nodes=node_order)

    def zero_grad(self, additional_tensors: list[Tensor] | None = None) -> None:
        """Clears the gradiens of all parameters that are optimized.

        Applied to all parameters in `self.params`.

        Args:
            additional_tensors (list[Tensor], optional): Extra Tensors for
            which to reset gradients. This could be activations that have
            explicitly set their `keep_grad` attribute to `True`, meaning
            they are not cleared before the backward pass of the graph
            they are part of. Defaults to None.
        """
        for param in chain(self.params, additional_tensors or []):
            param.grad = None

    @no_grad_fn
    @abstractmethod
    def step(self) -> None:
        """The step function to update the parameters.

        Must be implemented by the specific optimizer.
        """


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def __init__(
        self,
        params: list[Parameter],
        *,
        lr: float = 1e-3,
        friction: float = 1,
        weight_decay: float = 0,
    ):
        """The stochastic gradient descent optimizer.

        Note: By default, vanilla SGD is used. However, when setting
        the arguments accordingly, it can become SGD with momentum and also
        apply weight decay.

        **Standard SGD:** `friction=1, weight_decay=0`
        **SGD w/ momentum:** `friction<1, weight_decay=0`
        **SGDW:** `friction<1, weight_decay>0`

        Args:
            params (list[Parameter]): Parameters to optimize.
            lr (float, optional): The learing rate. Defaults to 1e-3.
            friction (float, optional): How much friction to apply on the
                momentum. If friction is 1 (100%), then we do not use momentum,
                as in every step all previous momentum is lost.
                If momentum is desired, set `friction<1`. A typical value is `0.1`,
                so 10% of momentum is lost every step due to friction.
                Defaults to 1.
            weight_decay (float, optional): Decay rate of the weights/parameters,
                equals the weight of L2-regularization on the loss.
                If SGDW is used, a typical value is `0.01`.
                Defaults to `0`.
        """
        super().__init__(params=params, lr=lr)

        if not 0 <= friction <= 1:
            raise ValueError(f"friction must be in [0, 1], got {friction}")

        self.m: list[Tensor] | None = (
            [zeros_like(p, dtype=p.dtype, requires_grad=False) for p in self.params]
            if friction < 1
            else None
        )

        self.friction = tensor(friction)
        self.weight_decay = tensor(weight_decay)

    @no_grad_fn
    def step(self) -> None:
        """Performs a single gradient descent step.

        Raises:
            ValueError: If a parameter has no gradient.
        """
        for idx, param in enumerate(self.params):
            if param.grad is None:
                raise ValueError("Gradient of parameter must not be None in step function")

            if self.m is not None:
                # we lose momentum through "friction", the momentum remaining from the previous step
                # is (1-self.friction)
                # we add param.grad as new force from the current position
                self.m[idx] = (1 - self.friction) * self.m[idx] + param.grad
                grad = self.m[idx]
            else:
                grad = param.grad

            param[...] = (
                (1 - self.lr * self.weight_decay) * param  # weight decay part
                - self.lr * grad  # gradient part
            )


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        params: list[Parameter],
        *,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0,
    ):
        """The Adam optimizer.

        Note: By setting `weight_decay` > 0 this becomes `AdamW`.

        Args:
            params (list[Parameter]): Parameters to optimize.
            lr (float, optional): The learning rate, also called `alpha`
                in the paper. Defaults to 1e-3.
            beta_1 (float, optional): Exponential decay rate for
                the momentum. Defaults to 0.9.
            beta_2 (float, optional): Exponential decay rate for
                the noise. Defaults to 0.999.
            epsilon (float, optional): Value added to `v` to improve
                numerical stability and avoid division by zero. Defaults to 1e-8.
            weight_decay (float, optional): Decay rate of the weights/parameters,
                equals the weight of L2-regularization on the loss.
                If greater than `0`, the optimizer becomes **AdamW**.
                If AdamW is used, a typical value is `0.01`.
                Defaults to `0`, meaning vanilla Adam is used.
        """
        super().__init__(params=params, lr=lr)
        self.m: list[Tensor] = [
            zeros_like(p, dtype=p.dtype, requires_grad=False) for p in self.params
        ]
        self.v: list[Tensor] = [
            zeros_like(p, dtype=p.dtype, requires_grad=False) for p in self.params
        ]
        self.beta_1 = tensor(beta_1)
        self.beta_2 = tensor(beta_2)
        self.epsilon = tensor(epsilon)
        self.t = tensor(0)
        self.weight_decay = tensor(weight_decay)

    @no_grad_fn
    def step(self) -> None:
        """Performs a single Adam step.

        Uses the slightly more efficient variant, which
        can be found at the end of section 2 in the paper: https://arxiv.org/pdf/1412.6980

        Raises:
            ValueError: If a parameter has no gradient.
        """
        self.t = tensor(self.t + 1)

        for idx, param in enumerate(self.params):
            if param.grad is None:
                raise ValueError("Gradient of parameter must not be None in step function")

            self.m[idx] = self.beta_1 * self.m[idx] + (1 - self.beta_1) * param.grad
            self.v[idx] = self.beta_2 * self.v[idx] + (1 - self.beta_2) * param.grad**2

            lr_t = self.lr * xp.sqrt(1 - self.beta_2**self.t) / (1 - self.beta_1**self.t)

            epsilon_hat = self.epsilon * xp.sqrt(1 - self.beta_2**self.t)

            # [...] -> in-place assignment
            param[...] = (
                (1 - self.lr * self.weight_decay) * param  # weight decay part
                - lr_t * self.m[idx] / (xp.sqrt(self.v[idx]) + epsilon_hat)  # gradient part
            )


__all__ = [
    "SGD",
    "Adam",
    "Optimizer",
    "toposort",
]
