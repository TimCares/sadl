<p align="center">
  <img src="assets/sadl_icon_light.png" alt="SADL Logo" width="200">
</p>

<h1 align="center">SADL: Simple Autograd Deep Learning</h1>

<p align="center">
  A minimal, actually readable deep learning framework built on NumPy and CuPy.<br>
  Automatic differentiation, neural network primitives, and optimization with just a handful of Python files.
</p>

<p align="center">
  <a href="https://pypi.org/project/py-sadl/"><img src="https://img.shields.io/pypi/v/py-sadl?style=flat" alt="PyPI version"></a>
  <a href="https://pypi.org/project/py-sadl/"><img src="https://img.shields.io/pypi/pyversions/py-sadl?style=flat" alt="Python versions"></a>
  <a href="https://pypi.org/project/py-sadl/"><img src="https://img.shields.io/pypi/l/py-sadl?style=flat" alt="License"></a>
  <a href="https://github.com/timcares/sadl/actions/workflows/ci_cd.yaml"><img src="https://github.com/timcares/sadl/actions/workflows/ci_cd.yaml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/timcares/sadl"><img src="https://codecov.io/gh/timcares/sadl/graph/badge.svg" alt="codecov"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/ruff-linted-blue?style=flat" alt="Ruff"></a>
  <a href="https://microsoft.github.io/pyright/"><img src="https://img.shields.io/badge/pyright-type%20checked-blue?style=flat" alt="Pyright"></a>
  <img alt="Docstring coverage" src="assets/interrogate_badge.png" height="22">
</p>

## The tale
Seek out [this](docs/ABOUT.md), to find out the story behind `SADL`.

## Demo
See [mnist_demo.ipynb](notebooks/mnist_demo.ipynb) for a working mini example of `sadl` on [mnist](https://huggingface.co/datasets/ylecun/mnist).

## Getting Started

A light description of all key components with examples: [GETTING_STARTED.md](docs/GETTING_STARTED.md)

## Installation

Using [uv](https://docs.astral.sh/uv/) for installation is recommended.

(I had to name the pypi project `py-sadl` instead of `sadl`, because `sadl` was too similar to an existing project.)

Install `uv` with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Install with uv (recommended)
uv add py-sadl

# With GPU support (CUDA 12.x)
uv add py-sadl --extra gpu
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
import numpy as np

# Create tensors
x = sadl.tensor([[1.0, 2.0], [3.0, 4.0]])

# Build a model
model = sadl.Mlp([
    sadl.Linear(dim_in=2, dim_out=4),
    sadl.ReLU(),
    sadl.Linear(dim_in=4, dim_out=1),
])

# Forward pass
output = model(x)
loss = np.sum(output)

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

SADL takes inspiration from both projects while pursuing its own path: building directly on NumPy's ndarray infrastructure.
By implementing NumPy's operator and dispatch protocols via `NDArrayOperatorsMixin`, `__array_ufunc__`, and `__array_function__`,
SADL achieves autograd without introducing a custom array backend.
This means existing NumPy code works unchanged, and the mental model stays close to the numerical computing patterns that researchers already know.

## Design Principles

### Build on NumPy

SADL builds on NumPy. This means:

- All NumPy operations work out of the box
- No need to learn a new tensor API
- Seamless interoperability with the scientific Python ecosystem
- Only a thin `Tensor` layer on top of NumPy arrays
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
- Device transfers (cpu ↔ cuda) are explicit operations

### Minimal but Complete

The framework includes only what is necessary for training neural networks:

- Tensor with autograd support
- Parameter for learnable weights
- Function base class for layers
- Optimizer base class with built-in SGD and Adam due to their frequent use
- Serialization for model persistence

Additional layers and optimizers can be built on these primitives without modifying core code.

### Key Components

**Tensor**: Wrapper around `np.ndarray` with additional attributes for autograd. Intercepts NumPy operations to build the computation graph.

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

The goal is to expand the ecosystem without losing the small and readable core.
There are definitely many more things worth adding, like more layers, architectures, and domain-specific tooling.
But if all of that keeps accumulating inside this repository, it creates exactly the kind of codebase bloat that makes larger frameworks hard to understand in the first place.

That is why the long-term direction is a small core `sadl` repository, plus separate plugin-style repositories for higher-level functionality, similar in spirit to projects like `timm`, `torchvision`, or `torchaudio`.
This keeps the core focused on tensors, gradients, backpropagation, functions, optimizers, and device handling, while allowing the surrounding ecosystem to grow independently.

### Built In
- Static graph compilation for repeated computations

### Separate Repositories
- Additional layers and components (convolution, batch normalization, attention)
- Higher-level model architectures and domain libraries
- XLA compilation backend for TPU support
- Automatic mixed precision training
- Distributed training primitives
