<p align="center">
  <img src="assets/sadl_icon_light.png" alt="SADL Logo" width="200">
</p>

<h1 align="center">SADL: Simple Autograd Deep Learning</h1>

<p align="center">
  A minimal, readable deep learning framework built on NumPy and CuPy.<br>
  Automatic differentiation, neural network primitives, and optimization in ~2000 lines of Python.
</p>

## Installation

Using [uv](https://docs.astral.sh/uv/) for installation is recommended.

(I had to name the pypi project `py-sadl` instead of `sadl`, because `sadl` was too similar to an existing project.)

```bash
# Install with uv (recommended)
uv add py-sadl

# With GPU support (CUDA 12.x)
uv add py-sadl --extra gpu

# With GPU support (CUDA 11.x)
uv add py-sadl --extra gpu-cuda11
```

Alternatively, using pip:

```bash
# Install with pip
pip install py-sadl

# With GPU support
pip install "py-sadl[gpu]"
```

## Quick Start

```python
import sadl

# Create tensors
x = sadl.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# Build a model
model = sadl.Mlp([
    sadl.Linear(dim_in=2, dim_out=4),
    sadl.ReLU(),
    sadl.Linear(dim_in=4, dim_out=1),
])

# Forward pass
output = model(x)
loss = output.sum()

# Backward pass and optimization
optimizer = sadl.SGD(list(model.parameters), lr=0.01)
optimizer.backward(loss)
optimizer.step()
optimizer.zero_grad()
```

## Motivation

Modern deep learning frameworks like PyTorch and TensorFlow are powerful but complex. Their codebases span millions of lines, making it difficult to understand how automatic differentiation and neural network training actually work at a fundamental level.

SADL addresses this by providing a complete, functional deep learning framework that remains small enough to read and understand in its entirety. Every component, from tensor operations to backpropagation, is implemented transparently using standard NumPy operations.

The goal is not to replace production frameworks, but to serve as an educational resource and a foundation for experimentation. Researchers and engineers can trace exactly how gradients flow through computations without navigating layers of abstraction.

## Related Projects

SADL joins a family of educational and minimal deep learning frameworks that have made autodiff more accessible:

**[micrograd](https://github.com/karpathy/micrograd)** by Andrej Karpathy is an elegant, minimal autograd engine operating on scalar values. In roughly 150 lines of code, it demonstrates the core concepts of backpropagation with remarkable clarity. micrograd is an excellent starting point for understanding how gradients flow through computations.

**[tinygrad](https://github.com/tinygrad/tinygrad)** by George Hotz takes a different approach, building a fully-featured deep learning framework with a focus on simplicity and hardware portability. tinygrad supports multiple backends and has grown into a serious alternative for running models on diverse hardware.

SADL takes inspiration from both projects while pursuing its own path: building directly on NumPy's ndarray infrastructure. By subclassing `numpy.ndarray` and intercepting operations via `__array_ufunc__` and `__array_function__`, SADL achieves autograd without introducing a new tensor abstraction. This means existing NumPy code works unchanged, and the mental model stays close to the numerical computing patterns that researchers already know.

## Design Principles

### Build on NumPy

SADL extends `numpy.ndarray` directly rather than wrapping arrays in custom containers. This means:

- All NumPy operations work out of the box
- No need to learn a new tensor API
- Seamless interoperability with the scientific Python ecosystem
- GPU support through CuPy with zero code changes

### Mathematical Functions as First-Class Citizens

Neural network layers are modeled as mathematical functions, matching how they appear in research papers. The `Function` abstract base class enforces a simple contract: implement `__call__` to define the forward pass. This creates a natural bridge between mathematical notation and code.

```python
class Sigmoid(Function):
    def __call__(self, x: Tensor) -> Tensor:
        return 1 / (xp.exp(-x) + 1)
```

### Explicit Over Implicit

SADL favors explicit behavior over magic:

- Gradients must be explicitly enabled with `requires_grad=True`
- Parameters are a distinct type that always tracks gradients
- The computation graph is visible and inspectable
- Device transfers are explicit operations

### Minimal but Complete

The framework includes only what is necessary for training neural networks:

- Tensor with autograd support
- Parameter for learnable weights
- Function base class for layers
- Optimizer base class with SGD implementation
- Serialization for model persistence

Additional layers and optimizers can be built on these primitives without modifying core code.

## How Autodiff Works

SADL implements reverse-mode automatic differentiation (backpropagation) using a dynamic computation graph, similar to PyTorch.

### The Computation Graph

In SADL, **Tensors are the computation graph**. There is no separate graph data structure. Each Tensor stores a `src` attribute pointing to the Tensors it was created from. This forms a back-referencing graph where each node knows its parents, but parents do not know their children:

```
Forward computation:

x ─┐
   ├─► z ─► loss
y ─┘

Graph structure (src references):

   loss
    │
    ▼
    z
   ╱ ╲
  ▼   ▼
  x   y
```

This is intentional. Deep learning frameworks optimize for backward traversal because that is what backpropagation requires. Starting from the loss, we follow `src` references backward through the graph to compute gradients. Forward references (parent to child) are unnecessary and would only consume memory.

### Forward Pass: Building the Graph

When operations are performed on Tensors with `requires_grad=True`, the graph builds itself automatically:

1. The `Tensor` class overrides `__array_ufunc__` and `__array_function__` to intercept NumPy operations
2. Each operation creates a new Tensor that stores:
  - `src`: References to input tensors (the parents in the graph)
  - `backward_fn`: The gradient function for this operation
  - `op_ctx`: Any context needed for gradient computation (axis, masks, etc.)
3. The graph grows dynamically as operations execute

```python
x = sadl.tensor([1.0, 2.0], requires_grad=True)  # leaf, src = ()
y = sadl.tensor([3.0, 4.0], requires_grad=True)  # leaf, src = ()
z = x * y      # z.src = (x, y), z.backward_fn = mul_backward
loss = z.sum() # loss.src = (z,), loss.backward_fn = sum_backward
```

A more complex example:

```
a = tensor(...)      # leaf
b = tensor(...)      # leaf
c = tensor(...)      # leaf

d = a + b            # d.src = (a, b)
e = d * c            # e.src = (d, c)
f = e.sum()          # f.src = (e,)

Graph (following src backwards from f):

    f
    │
    ▼
    e
   ╱ ╲
  ▼   ▼
  d   c
 ╱ ╲
▼   ▼
a   b
```

### Backward Pass: Computing Gradients

When `optimizer.backward(loss)` is called:

1. **Topological Sort**: The graph is traversed from the loss tensor to find all nodes, ordered so that each node appears after all nodes that depend on it. This uses an iterative stack-based algorithm to avoid recursion limits on deep graphs.
2. **Gradient Propagation**: Starting from the loss (seeded with gradient 1.0), each node's `backward_fn` is called with:
  - The input tensors (`src`)
  - Which inputs need gradients (`compute_grad`)
  - The upstream gradient (`grad_out`)
  - Operation context (`op_ctx`)
3. **Gradient Accumulation**: Gradients flow backward through the graph. When a tensor is used in multiple operations, gradients are summed.
4. **Graph Cleanup**: After backpropagation, the graph structure is cleared to free memory. Parameter gradients are retained for the optimizer step.

### Gradient Operations Registry

Each supported operation has a corresponding backward function registered in `grad_ops.py`:

```python
@register_grad_op
@broadcastable
def mul_backward(*inputs, compute_grad, grad_out):
    x, y = inputs
    grad_x = y * grad_out if compute_grad[0] else None
    grad_y = x * grad_out if compute_grad[1] else None
    return grad_x, grad_y
```

The `@broadcastable` decorator handles gradient reduction when inputs were broadcast during the forward pass.

### Supported Operations

Unary: `abs`, `negative`, `sqrt`, `square`, `exp`, `log`, `sin`, `cos`

Binary: `add`, `subtract`, `multiply`, `divide`, `power`, `matmul`, `maximum`, `minimum`

Reductions: `sum`, `mean`, `max`, `min`

## Architecture

```
sadl/
├── __init__.py     # Public API re-exports
├── backend.py      # NumPy/CuPy abstraction
├── disk.py         # Saving and loading data to/from disk
├── tensor.py       # Tensor, Parameter, serialization
├── grad_ops.py     # Gradient operation registry
├── function.py     # Neural network layers
├── optimizer.py    # Optimizer base class, SGD, backpropagation
├── ops.py          # Array creation and device utilities
└── utils.py        # Device transfer utilities
```

### Key Components

**Tensor**: Subclass of `numpy.ndarray` with additional attributes for autograd. Intercepts NumPy operations to build the computation graph.

**Parameter**: Tensor subclass for learnable weights. Always requires gradients and retains them after backward pass for gradient accumulation.

**Function**: Abstract base class for neural network layers. Provides parameter traversal, device management, and train/inference mode switching.

**Optimizer**: Abstract base class that owns the backward pass. Performs topological sort, gradient computation, and graph cleanup.

**GradOp Registry**: Dictionary mapping operation names to backward functions. New operations can be registered with a decorator.

## Serialization

SADL uses a custom binary format (`.sadl` files) for efficient tensor storage:

- 4-byte magic header for format validation
- Version byte for forward compatibility
- Compact encoding of dtype, shape, and raw data
- Support for single tensors or ordered dictionaries of tensors

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, commands, and guidelines.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for behavior guidelines. The file was created using the [Contributor Covenant](https://www.contributor-covenant.org).

## Future Plans

- Static graph compilation for repeated computations
- Additional layers and components (convolution, batch normalization, attention)
- More optimizers (Adam, AdamW, RMSprop)
- XLA compilation backend for TPU support
- Automatic mixed precision training
- Distributed training primitives

## See Also

- `docs/API_REFERENCE.md`: Complete API documentation
