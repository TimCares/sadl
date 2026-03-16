# Function

In `SADL`, every component of a Neural Network is represented as a `Function`.

That name is on purpose. A linear layer is a mathematical function. An activation like ReLU is a mathematical function. A full model composed of several layers is also a mathematical function. The code mirrors how these components appear in research papers.

The key idea is simple: a `Function` has parameters, and it has a `__call__` method that defines the forward pass.

## `__call__` As the Forward Pass

Every `Function` subclass must implement `__call__`. This is where the actual computation happens.

```python
import sadl
import numpy as np

class MyLinear(sadl.Function):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        self.W = sadl.Parameter(np.random.randn(dim_in, dim_out).astype(np.float32))
        self.b = sadl.Parameter(np.zeros((dim_out,), dtype=np.float32))

    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:
        return x @ self.W + self.b
```

That is all it takes. The parameters are stored as attributes, and `__call__` uses them.
When the client writes `output = model(x)`, Python calls `__call__`, which builds the computation graph that backpropagation later walks.

This is the contract that makes everything work: parameters that should be optimized must participate in the forward pass. Otherwise backpropagation cannot reach them, and the optimizer will skip them.

## Parameters As Attributes

A `Function` discovers its parameters through attribute traversal. There is no explicit registration step like `nn.Module.register_parameter` in PyTorch. If an attribute is a `Parameter`, the `Function` finds it.

Parameters can live in several places:

- **Direct attributes**: `self.W = sadl.Parameter(...)`
- **Lists**: `self.weights = [sadl.Parameter(...), sadl.Parameter(...)]`
- **Dicts**: `self.lookup = {"key": sadl.Parameter(...)}`
- **Nested Functions**: `self.layer = sadl.Linear(dim_in=3, dim_out=4)` (its parameters are found recursively)

There is one restriction: parameters and nested Functions must be stored in **mutable containers**.
Tuples and sets are not allowed.

Tuples are immutable, which means operations like `copy_to_device` cannot replace a parameter inside a tuple with its copy on the new device. Sets have no stable ordering, making parameter access unpredictable.

If you try it, `SADL` will raise a `TypeError` and explain why.

## `get_parameters` and Naming

`get_parameters()` recursively collects all parameters into an `OrderedDict[str, Parameter]`, keyed by their path through the attribute hierarchy.

```python
model = sadl.Mlp([
    sadl.Linear(dim_in=784, dim_out=128),
    sadl.ReLU(),
    sadl.Linear(dim_in=128, dim_out=10),
])

for name, param in model.get_parameters().items():
    print(name, param.shape)

# layers[0].W (784, 128)
# layers[0].b (128,)
# layers[2].W (128, 10)
# layers[2].b (10,)
```

Note that `layers[1]` (the ReLU) has no parameters, so nothing appears for it.

These names are stable as long as the model structure does not change. They are the same names the optimizer uses to key its per-parameter state, and the same names used for serialization. Everything agrees on one naming scheme.

The `parameters` property is a shorthand that returns just the values, without the names:

```python
for param in model.parameters:
    print(param.shape)
```

## Moving to a Device

`copy_to_device` moves all parameters to the specified device:

```python
gpu = sadl.TensorDevice("cuda", device_id=0)
model = model.copy_to_device(gpu)
```

Under the hood, this traverses all parameters and replaces each one with a copy on the new device. That is why mutable containers are required: the traversal needs to write back the new `Parameter` object.

The method returns `self` for chaining.

## Loading and Saving Parameters

`get_parameters()` returns an `OrderedDict[str, Parameter]` that can be saved to disk with `sadl.save` and later loaded back with `sadl.load`.

To load parameters back into a model with the same architecture:

```python
model.load_parameters(state, match_function_device=True)
```

`load_parameters` matches keys from the provided dict to the model's parameter paths. If a key is missing and `partial=False` (the default), it raises.

Setting `match_function_device=True` automatically copies loaded parameters to whatever device the model is currently on. Without it, a device mismatch raises an error.

An important detail: `load_parameters` only replaces the **data buffer** of each parameter, not the `Parameter` object itself. This preserves the existing object identity and any references to it (e.g. in an optimizer).

## Training and Inference Mode

Some layers behave differently during training and inference. Dropout randomly zeros activations during training but passes everything through during inference. BatchNorm uses running statistics at inference time.

`Function` supports this through `train()` and `inference()`, which set the `is_training` flag on all parameters:

```python
model.train()       # training mode
model.inference()   # inference mode
```

The `is_training` property reports whether all parameters are currently in training mode.

## Freezing

To freeze a function (prevent gradient computation for all its parameters):

```python
model.requires_grad = False
```

This sets `requires_grad = False` on every parameter. The getter returns `True` only if **all** parameters require a gradient; a single frozen parameter makes the whole function report `False`.

## Built-In Functions

`SADL` provides a small set of built-in functions. The set is intentionally kept minimal to avoid bloating the codebase.

**Layers with parameters:**
- `Linear(dim_in, dim_out, bias=True, dtype=None)`: Dense layer with Xavier initialization.
- `Mlp(layers)`: Sequential composition of functions.

**Activation functions (no parameters):**
- `Sigmoid`
- `Softmax`
- `LogSoftmax` (fused `log(softmax(x))` for numerical stability)
- `ReLU`

A function does not need to have parameters. Activation functions are a natural example: they transform input but have no learnable weights. They are still `Function` subclasses because they represent mathematical functions that participate in the computation graph.

## Properties

- `parameters`: View of all Parameters.
- `requires_grad`: Get/set gradient tracking for all parameters.
- `device`: Tuple of devices where parameters reside (see [`DEVICE.md`](DEVICE.md) for why it is a tuple).
- `is_training`: Whether the model is in training mode.

## Methods

- `__call__(...)`: Forward pass (abstract, must be implemented).
- `get_parameters(to_device=None)`: Get `OrderedDict` of named parameters.
- `load_parameters(state, match_function_device, partial)`: Load parameter values from a dict.
- `copy_to_device(device)`: Move all parameters to a device.
- `traverse_parameters(on_parameter)`: Low-level traversal hook.
- `train()`: Set to training mode.
- `inference()`: Set to inference mode.

## Summary

`Function` is deliberately simple. It is an abstract class with `__call__`, automatic parameter discovery, and a handful of utilities for device management, serialization, and training mode.

That is enough to build anything from a single linear layer to a full transformer.

Further reading:
- What a Parameter actually is -> [`TENSOR.md`](TENSOR.md)
- How the computation graph is built from the forward pass -> [`autograd/COMPUTATION_GRAPH.md`](autograd/COMPUTATION_GRAPH.md)
- How the optimizer uses `get_parameters()` -> [`OPTIMIZER.md`](OPTIMIZER.md)
