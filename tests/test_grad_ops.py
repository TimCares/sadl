"""Tests for gradient operations using finite difference verification.

Uses batched element-wise perturbation for efficient gradient checking.
Small tensor shapes are used due to O(n^2) memory from batching.

Test configuration is embedded directly in the grad_ops registry via GradOpSpec,
so adding a new op requires providing test metadata at registration time.

Note: Usually we should follow strict coding guidelines through ruff, but since this
is just testing code, we can make some exceptions. Hence the "noqa: ..." directives.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from sadl import Tensor, set_global_grad_mode, tensor, xp
from sadl.grad_ops import (
    _GRAD_OPS_REGISTRY,
    GradOpSpec,
    OpInputs,
    OpType,
)

# Disable gradient tracking globally for tests
# (backward functions use numpy ops that would create Tensors with grad tracking)
set_global_grad_mode(enabled=False)


def shift_axis(
    axis: int | tuple[int, ...] | None,
    shift: int = 1,
) -> tuple[int, ...] | None:
    """Shift axis indices to account for an added batch dimension.

    Args:
        axis (int | tuple[int, ...] | None): Original axis specification.
        shift (int): Amount to shift each axis index.

    Returns:
        tuple[int, ...] | None: Shifted axes as tuple, or None if axis was None.
    """
    if axis is None:
        return None
    if isinstance(axis, int):
        return (axis + shift,)
    return tuple(a + shift for a in axis)


def make_batched_op(
    op: Callable[..., xp.ndarray],
    *fixed_args: Any,
    is_reduction: bool = True,
    axis: int | tuple[int, ...] | None = None,
    **fixed_kwargs: Any,
) -> Callable[[xp.ndarray], xp.ndarray]:
    """Create a batched operation for finite difference computation.

    Handles axis shifting for reductions and final sum to scalar per batch element.

    Args:
        op (Callable[..., xp.ndarray]): The numpy operation (e.g. xp.sum, xp.matmul).
        *fixed_args (Any): Additional positional args passed to op.
        is_reduction (bool): Whether op accepts an axis argument. If True, axis is
            shifted by +1. If False, op is called without axis.
        axis (int | tuple[int, ...] | None): Original axis specification.
            If None and is_reduction=True, reduces all dims except batch.
        **fixed_kwargs (Any): Additional kwargs passed to op.

    Returns:
        Callable[[xp.ndarray], xp.ndarray]: Function mapping (n_elements, *shape)
            to (n_elements,).
    """
    shifted_axis = shift_axis(axis)

    def batched(x: xp.ndarray) -> xp.ndarray:
        if is_reduction:
            effective_axis = shifted_axis if shifted_axis is not None else tuple(range(1, x.ndim))
            result = op(x, *fixed_args, axis=effective_axis, **fixed_kwargs)
        else:
            result = op(x, *fixed_args, **fixed_kwargs)

        # Sum over remaining non-batch dims to get (n_elements,)
        if result.ndim > 1:
            reduce_axes = tuple(range(1, result.ndim))
            result = xp.sum(result, axis=reduce_axes)

        return result

    return batched


def fd_gradient_batched(
    x: Tensor,
    batched_func: Callable[[xp.ndarray], xp.ndarray],
    eps: float = 1e-5,
) -> xp.ndarray:
    """Compute gradient via batched centered finite differences.

    Fully vectorized with no Python loops. Uses O(n^2) memory where n = x.size.

    Args:
        x (Tensor): Input tensor of shape (*shape).
        batched_func (Callable[[xp.ndarray], xp.ndarray]): Function mapping
            (n_elements, *shape) to (n_elements,).
        eps (float): Perturbation size for finite differences.

    Returns:
        xp.ndarray: Gradient array with same shape as x.
    """
    shape = x.shape
    n_elements = int(xp.prod(shape))

    # Create all perturbations
    modifier = (xp.eye(n_elements) * eps).reshape((n_elements, *shape))

    # Batched forward passes
    x_plus = x.data + modifier
    x_minus = x.data - modifier

    # Apply batched function
    f_plus = batched_func(x_plus)
    f_minus = batched_func(x_minus)

    # Centered difference
    grad_flat = (f_plus - f_minus) / (2 * eps)

    return grad_flat.reshape(shape)


def generate_input(
    shape: tuple[int, ...],
    constraint: str | None = None,
    rng: np.random.Generator | None = None,
) -> xp.ndarray:
    """Generate random input data respecting constraints.

    Args:
        shape (tuple[int, ...]): Shape of the array to generate.
        constraint (str | None): Constraint type. One of:
            None: uniform(-2.0, 2.0)
            "positive": uniform(0.1, 2.0)
            "nonzero": uniform(0.5, 2.0), away from zero
        rng (np.random.Generator | None): Random number generator.

    Returns:
        xp.ndarray: Random array of the specified shape.
    """
    if rng is None:
        rng = np.random.default_rng()

    if constraint == "positive":
        return rng.uniform(0.1, 2.0, shape)
    elif constraint == "nonzero":
        data = rng.uniform(0.5, 2.0, shape)  # all positive, away from zero
        return data
    else:
        return rng.uniform(-2.0, 2.0, shape)


def get_numpy_op(spec: GradOpSpec) -> Callable[..., xp.ndarray]:
    """Get the numpy operation from the spec's forward_names.

    Tries each name in forward_names until one is found in numpy.

    Args:
        spec (GradOpSpec): Gradient operation specification.

    Returns:
        Callable[..., xp.ndarray]: The numpy function.

    Raises:
        ValueError: If no name in forward_names exists in numpy.
    """
    for name in spec.forward_names:
        if hasattr(xp, name):
            # cast to make mypy happy
            return cast(Callable[..., xp.ndarray], getattr(xp, name))
    raise ValueError(f"No numpy op found for forward_names={spec.forward_names}")


# =============================================================================
# Test Implementation
# =============================================================================


def _get_unique_op_names() -> list[str]:
    """Get unique operation names from registry, avoiding aliases.

    E.g. mul_backward is available via the keys `mul` and `multiply`,
    so we deduplicate by identity.

    Returns:
        list[str]: List of canonical operation names (one per GradOpSpec).
    """
    seen_specs: set[int] = set()
    unique_names: list[str] = []
    for name, spec in _GRAD_OPS_REGISTRY.items():
        spec_id = id(spec)
        if spec_id not in seen_specs:
            seen_specs.add(spec_id)
            unique_names.append(name)
    return unique_names


@pytest.mark.parametrize("op_name", _get_unique_op_names())
def test_grad_op(op_name: str) -> None:
    """Test gradient operation against finite differences.

    Configuration is embedded in GradOpSpec, so no separate config dict is needed.

    Args:
        op_name (str): Name of the operation to test.
    """
    # Get spec from registry (config is embedded in the spec)
    spec: GradOpSpec = _GRAD_OPS_REGISTRY[op_name]

    # Check if skipped
    if spec.skip_test:
        pytest.skip(f"Skipped: {spec.skip_reason}")

    # Extract config from spec
    n_inputs = spec.op_inputs.value
    is_reduction = spec.op_type == OpType.REDUCTION
    is_matmul = spec.op_type == OpType.LINALG
    constraints = spec.constraints or {}

    # Use fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)

    # Small shapes due to O(n^2) memory from batched perturbation
    if is_matmul:
        shape_x = (3, 4)
        shape_y = (4, 2)
    else:
        shape_x = (3, 4)
        shape_y = (3, 4)

    # Generate inputs
    x_data = generate_input(shape_x, constraints.get("x"), rng)
    x = tensor(x_data, dtype=xp.float64)

    if n_inputs >= OpInputs.BINARY.value:
        y_data = generate_input(shape_y, constraints.get("y"), rng)
        y = tensor(y_data, dtype=xp.float64)

    # Get the backward function
    grad_op = spec.backward_fn

    # Parameters for comparison
    eps = 1e-5
    rtol = 1e-4
    atol = 1e-6

    # Run appropriate test based on operation type
    if spec.op_inputs == OpInputs.UNARY:
        _test_unary_op(spec, x, grad_op, is_reduction, eps, rtol, atol)
    elif is_matmul:
        _test_matmul_op(x, y, grad_op, eps, rtol, atol)
    else:
        _test_binary_op(spec, x, y, grad_op, is_reduction, eps, rtol, atol)


def _test_unary_op(  # noqa: PLR0913
    spec: GradOpSpec,
    x: Tensor,
    grad_op: Callable[..., tuple[xp.ndarray | None, ...]],
    is_reduction: bool,
    eps: float,
    rtol: float,
    atol: float,
) -> None:
    """Test a unary gradient operation against finite differences.

    Args:
        spec (GradOpSpec): Operation specification.
        x (Tensor): Input tensor.
        grad_op (Callable): Backward function to test.
        is_reduction (bool): Whether the op is a reduction.
        eps (float): Finite difference perturbation size.
        rtol (float): Relative tolerance for comparison.
        atol (float): Absolute tolerance for comparison.
    """
    numpy_op = get_numpy_op(spec)

    # Compute forward pass to get output shape for grad_out
    output = numpy_op(x.data)
    grad_out = xp.ones_like(output)

    # Analytical gradient, backward functions expect Tensors for .shape/.ndim access
    analytical_grad = grad_op(x, compute_grad=(True,), grad_out=grad_out)[0]

    # Finite difference gradient
    if is_reduction:
        batched_fn = make_batched_op(numpy_op, is_reduction=True)
    else:
        # Element-wise: apply op then sum for scalar loss
        def elem_wise_fn(arr: xp.ndarray) -> xp.ndarray:
            result = numpy_op(arr)
            reduce_axes = tuple(range(1, result.ndim))
            return xp.sum(result, axis=reduce_axes)

        batched_fn = elem_wise_fn

    fd_grad = fd_gradient_batched(x, batched_fn, eps=eps)

    # Convert to numpy arrays for comparison
    analytical_np = np.asarray(analytical_grad)
    fd_np = np.asarray(fd_grad)

    assert np.allclose(analytical_np, fd_np, rtol=rtol, atol=atol), (
        f"Gradient mismatch for {spec.forward_names[0]}:\n"
        f"Analytical:\n{analytical_np}\n"
        f"FD:\n{fd_np}\n"
        f"Max diff: {np.max(np.abs(analytical_np - fd_np))}"
    )


def _test_binary_op(  # noqa: PLR0913
    spec: GradOpSpec,
    x: Tensor,
    y: Tensor,
    grad_op: Callable[..., tuple[xp.ndarray | None, ...]],
    is_reduction: bool,
    eps: float,
    rtol: float,
    atol: float,
) -> None:
    """Test a binary gradient operation against finite differences.

    Args:
        spec (GradOpSpec): Operation specification.
        x (Tensor): First input tensor.
        y (Tensor): Second input tensor.
        grad_op (Callable): Backward function to test.
        is_reduction (bool): Whether the op is a reduction.
        eps (float): Finite difference perturbation size.
        rtol (float): Relative tolerance for comparison.
        atol (float): Absolute tolerance for comparison.
    """
    numpy_op = get_numpy_op(spec)
    op_name = spec.forward_names[0]

    # Compute forward pass
    output = numpy_op(x.data, y.data)
    grad_out = xp.ones_like(output)

    # Analytical gradients - backward functions expect Tensors
    analytical_grad_x, analytical_grad_y = grad_op(
        x, y, compute_grad=(True, True), grad_out=grad_out
    )

    # Finite difference gradient w.r.t. x (y fixed)
    if is_reduction:
        batched_fn_x = make_batched_op(
            lambda arr: numpy_op(arr, y.data),
            is_reduction=True,
        )
    else:

        def elem_wise_fn_x(arr: xp.ndarray) -> xp.ndarray:
            result = numpy_op(arr, y.data)
            reduce_axes = tuple(range(1, result.ndim))
            return xp.sum(result, axis=reduce_axes)

        batched_fn_x = elem_wise_fn_x

    fd_grad_x = fd_gradient_batched(x, batched_fn_x, eps=eps)

    # Finite difference gradient w.r.t. y (x fixed)
    if is_reduction:
        batched_fn_y = make_batched_op(
            lambda arr: numpy_op(x.data, arr),
            is_reduction=True,
        )
    else:

        def elem_wise_fn_y(arr: xp.ndarray) -> xp.ndarray:
            result = numpy_op(x.data, arr)
            reduce_axes = tuple(range(1, result.ndim))
            return xp.sum(result, axis=reduce_axes)

        batched_fn_y = elem_wise_fn_y

    fd_grad_y = fd_gradient_batched(y, batched_fn_y, eps=eps)

    # Convert to numpy arrays for comparison
    analytical_x_np = np.asarray(analytical_grad_x)
    analytical_y_np = np.asarray(analytical_grad_y)
    fd_x_np = np.asarray(fd_grad_x)
    fd_y_np = np.asarray(fd_grad_y)

    # Compare x gradient
    assert np.allclose(analytical_x_np, fd_x_np, rtol=rtol, atol=atol), (
        f"Gradient mismatch for {op_name} w.r.t. x:\n"
        f"Analytical:\n{analytical_x_np}\n"
        f"FD:\n{fd_x_np}\n"
        f"Max diff: {np.max(np.abs(analytical_x_np - fd_x_np))}"
    )

    # Compare y gradient
    assert np.allclose(analytical_y_np, fd_y_np, rtol=rtol, atol=atol), (
        f"Gradient mismatch for {op_name} w.r.t. y:\n"
        f"Analytical:\n{analytical_y_np}\n"
        f"FD:\n{fd_y_np}\n"
        f"Max diff: {np.max(np.abs(analytical_y_np - fd_y_np))}"
    )


def _test_matmul_op(  # noqa: PLR0913
    A: Tensor,  # noqa: N803
    B: Tensor,  # noqa: N803
    grad_op: Callable[..., tuple[xp.ndarray | None, ...]],
    eps: float,
    rtol: float,
    atol: float,
) -> None:
    """Test matrix multiplication gradient against finite differences.

    Args:
        A (Tensor): First matrix (shape i x j).
        B (Tensor): Second matrix (shape j x k).
        grad_op (Callable): Backward function to test.
        eps (float): Finite difference perturbation size.
        rtol (float): Relative tolerance for comparison.
        atol (float): Absolute tolerance for comparison.
    """
    # Forward pass
    Z = xp.matmul(A.data, B.data)
    grad_out = xp.ones_like(Z)

    # Analytical gradients - backward functions expect Tensors
    analytical_grad_A, analytical_grad_B = grad_op(
        A, B, compute_grad=(True, True), grad_out=grad_out
    )

    # FD gradient w.r.t. A (B fixed)
    # A_batched @ B: (n, i, j) @ (j, k) -> (n, i, k) -> sum to (n,)
    batched_fn_A = make_batched_op(xp.matmul, B.data, is_reduction=False)
    fd_grad_A = fd_gradient_batched(A, batched_fn_A, eps=eps)

    # FD gradient w.r.t. B (A fixed)
    # Need A @ B_batched: use einsum 'ij,njk->nik'
    batched_fn_B = make_batched_op(
        lambda arr, a: xp.einsum("ij,njk->nik", a, arr),
        A.data,
        is_reduction=False,
    )
    fd_grad_B = fd_gradient_batched(B, batched_fn_B, eps=eps)

    # Convert to numpy arrays for comparison
    analytical_A_np = np.asarray(analytical_grad_A)
    analytical_B_np = np.asarray(analytical_grad_B)
    fd_A_np = np.asarray(fd_grad_A)
    fd_B_np = np.asarray(fd_grad_B)

    # Compare A gradient
    assert np.allclose(analytical_A_np, fd_A_np, rtol=rtol, atol=atol), (
        f"Gradient mismatch for matmul w.r.t. A:\n"
        f"Analytical:\n{analytical_A_np}\n"
        f"FD:\n{fd_A_np}\n"
        f"Max diff: {np.max(np.abs(analytical_A_np - fd_A_np))}"
    )

    # Compare B gradient
    assert np.allclose(analytical_B_np, fd_B_np, rtol=rtol, atol=atol), (
        f"Gradient mismatch for matmul w.r.t. B:\n"
        f"Analytical:\n{analytical_B_np}\n"
        f"FD:\n{fd_B_np}\n"
        f"Max diff: {np.max(np.abs(analytical_B_np - fd_B_np))}"
    )
