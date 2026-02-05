# SADL API Reference

## Quick Start

```python
# When installed as standalone package:
import sadl

# When used within the monorepo:
from src.lib import sadl

# Create tensors
x = sadl.tensor([[1.0, 2.0]], requires_grad=True)

# Build model
model = sadl.Mlp([
    sadl.Linear(dim_in=2, dim_out=4),
    sadl.ReLU(),
    sadl.Linear(dim_in=4, dim_out=1),
])

# Train
optimizer = sadl.SGD(list(model.parameters), lr=0.01)
output = model(x)
loss = output.sum()
optimizer.backward(loss)
optimizer.step()
optimizer.zero_grad()
```

## Backend

| Export | Description |
|--------|-------------|
| `BACKEND` | Current backend: `"numpy"` or `"cupy"` |
| `TensorDevice` | Type alias: `Literal["cpu"] \| int` |

## Tensor Classes

### Tensor

Main array class with autograd support. Subclass of `numpy.ndarray`.

**Attributes:**
- `requires_grad`: Whether gradients are tracked
- `grad`: Stored gradient after backward pass
- `src`: Parent tensors in computation graph
- `keep_grad`: Whether to retain gradient after backward

**Methods:**
- `copy_to_device(device)`: Move to CPU or GPU
- `detach(in_place=False)`: Remove from computation graph
- `cpu()`: Move to CPU
- `gpu(device_id=0)`: Move to GPU
- `is_leaf()`: True if no parent tensors

### Parameter

Tensor subclass for learnable weights. Always has `requires_grad=True` and `keep_grad=True`.

**Additional attributes:**
- `is_training`: Controls behavior of layers like Dropout

## Factory Functions

| Function | Description |
|----------|-------------|
| `tensor(data, *, dtype, device, requires_grad, keep_grad)` | Create tensor on specified device |
| `ones_like(other, *, dtype, requires_grad)` | Tensor of ones matching shape |
| `zeros_like(other, *, dtype, requires_grad)` | Tensor of zeros matching shape |

## Context Managers

| Export | Description |
|--------|-------------|
| `no_grad` | Context manager disabling gradient tracking |
| `no_grad_fn` | Decorator disabling gradients in a function |

```python
with sadl.no_grad():
    y = model(x)  # No graph built

@sadl.no_grad_fn
def inference(model, x):
    return model(x)
```

## Serialization

| Function | Description |
|----------|-------------|
| `save(data, file_path)` | Save tensor(s) to `.sadl` file |
| `load(file_path)` | Load tensor(s) from `.sadl` file |

Supports single tensors or `OrderedDict[str, Tensor]`.

## Neural Network Layers

### Function (ABC)

Base class for all layers. Subclasses must implement `__call__`.

**Properties:**
- `parameters`: View of all Parameter objects
- `requires_grad`: Get/set gradient tracking for all parameters
- `device`: Tuple of devices where parameters reside

**Methods:**
- `get_parameters(to_device=None)`: Get OrderedDict of parameters
- `load_parameters(parameters, match_function_device, partial)`: Load parameter values
- `copy_to_device(device)`: Move all parameters to device
- `train()`: Set training mode
- `inference()`: Set inference mode

### Built-in Layers

| Layer | Signature | Description |
|-------|-----------|-------------|
| `Linear` | `Linear(dim_in, dim_out, bias=True, dtype=float32)` | Fully connected layer |
| `ReLU` | `ReLU()` | ReLU activation |
| `Sigmoid` | `Sigmoid()` | Sigmoid activation |
| `Mlp` | `Mlp(layers)` | Sequential container |

## Optimizers

### Optimizer (ABC)

Base class for optimizers. Owns the backward pass.

**Properties:**
- `state`: View of optimizer state tensors
- `device`: Tuple of devices where state resides

**Methods:**
- `backward(loss)`: Compute gradients via backpropagation
- `step()`: Update parameters (abstract)
- `zero_grad(additional_tensors=None)`: Clear gradients
- `get_state(to_device=None)`: Get OrderedDict of state
- `load_state(state, match_device, partial)`: Load state
- `copy_to_device(device)`: Move state to device

### SGD

Stochastic Gradient Descent optimizer.

```python
optimizer = sadl.SGD(params, lr=1e-3)
```

## Gradient Operations

The `grad_ops` module provides access to the gradient registry.

| Export | Description |
|--------|-------------|
| `get_grad_op(name)` | Get backward function by operation name |
| `get_grad_op_spec(name)` | Get full `GradOpSpec` with metadata |
| `register_grad_op` | Decorator factory to register backward function with metadata |
| `OpType` | Enum: `ELEMENTWISE`, `REDUCTION`, `MOVEMENT`, `LINALG` |
| `OpInputs` | Enum: `UNARY` (1), `BINARY` (2), `TERNARY` (3) |
| `GradOpSpec` | Dataclass holding backward function and metadata |

**Supported operations:**

- Unary: `abs`, `negative`, `sqrt`, `square`, `exp`, `log`, `sin`, `cos`
- Binary: `add`, `subtract`, `mul`, `div`, `power`, `matmul`, `maximum`, `minimum`
- Reductions: `sum`, `mean`, `max`, `min`

### Custom Gradient Example

```python
from sadl.grad_ops import register_grad_op, OpType, OpInputs

@register_grad_op(
    op_type=OpType.ELEMENTWISE,
    op_inputs=OpInputs.UNARY,
)
def my_op_backward(*inputs, compute_grad, grad_out, **kwargs):
    x = inputs[0]
    grad_x = grad_out * 2 if compute_grad[0] else None
    return (grad_x,)
```
