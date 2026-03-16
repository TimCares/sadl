# Optimizer

The `Optimizer` in `SADL` performs the actual optimization of model parameters.
That sounds obvious, but there is one important thing to know:

The connection between `backward(loss)` and `step()` is quite indirect.

`backward(loss)` walks the computation graph starting at `loss` and writes gradients into `Tensor.grad`.
`step()` then iterates over the parameters passed to the optimizer and updates them using those gradients.

So the place where both worlds meet is simply:

```python
param.grad
```

That is the key idea of this page.

## The Training Pattern

This is why training code in `SADL` typically looks like this:

```python
# Create an optimizer
optimizer = sadl.SGD(model.get_parameters(), lr=0.01)
optimizer = optimizer.copy_to_device(gpu)

# A single training step
output = model(x)
loss = output.sum()
optimizer.backward(loss)
optimizer.step()
optimizer.zero_grad()
```

This order is not accidental.

1. `model(x)` builds a computation graph.
2. `loss` becomes the root of that graph.
3. `optimizer.backward(loss)` walks backwards through that graph and writes gradients into the participating parameters.
4. `optimizer.step()` reads those gradients and updates the tracked parameters.
5. `optimizer.zero_grad()` clears them for the next step.

## The Important Condition

If a parameter should be optimized, it must appear in the computation graph of which `loss` is the root.

Otherwise, backpropagation will never reach it.
`.grad` will stay `None`.
And `step()` will simply skip that parameter, since it only updates parameters that have a gradient.

So one can say:

- `backward(loss)` does not directly know which parameters the optimizer was constructed with
- `step()` does not directly know how the loss was computed
- the connection is established only through the graph and the `.grad` fields of the parameters

That is why the client has to ensure that the parameters passed to the optimizer actually participate in computing the loss.

## Why `Function.__call__` Matters

If the client follows the `SADL` philosophy and represents a model as a `Function`, then trainable parameters should participate in the forward pass defined by `__call__`.

That is exactly what makes this work:

```python
import sadl

class MyLinear(sadl.Function):
    def __init__(self):
        self.W = sadl.Parameter(...)
        self.b = sadl.Parameter(...)

    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:
        return x @ self.W + self.b
```

Now both `self.W` and `self.b` contribute to the output, therefore they also contribute to the loss, therefore backpropagation can write their gradients.

This is the clean path:

```python
output = model(x)
loss = output.sum()
optimizer.backward(loss)
optimizer.step()
```

## What Goes Wrong If a Parameter Is Unused

If a parameter is part of the optimizer, but does not participate in the computation of the loss, it cannot be optimized.

For example:

```python
import numpy as np
import sadl

class BadLinear(sadl.Function):
    def __init__(self):
        self.W = sadl.Parameter(np.ones((3, 3), dtype=np.float32))
        self.unused = sadl.Parameter(np.ones((3, 3), dtype=np.float32))

    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:
        return x @ self.W

model = BadLinear()
optimizer = sadl.SGD(model.get_parameters(), lr=1e-2)

x = sadl.tensor([[1.0, 2.0, 3.0]])
loss = model(x).sum()

optimizer.backward(loss)
optimizer.step()  # "unused" is silently skipped, no gradient to use
```

Here `self.unused` is a real `Parameter`, so it is included in `model.get_parameters()` and therefore tracked by the optimizer.
But it is never used in `__call__`, so it never appears in the graph rooted at `loss`.
That means backpropagation never reaches it.

The optimizer does **not** raise an error in this case. It simply skips the parameter, because `step()` only updates parameters that are both trainable and have a non-`None` gradient.

This is why the forward pass and the optimizer parameter list must agree.

## Freezing Parameters

To freeze parameters, simply set their `requires_grad` attribute to `False`.

The optimizer handles frozen parameters gracefully: `step()` skips any parameter whose `requires_grad` is `False` or whose `grad` is `None`. No error is raised.

Per-parameter state (like momentum in SGD or running averages in Adam) is created **lazily**, on the first `step()` call where the parameter actually gets updated. This means:

- Frozen parameters never allocate state, saving memory.
- If a parameter is later unfrozen (by setting `requires_grad = True`), the optimizer creates its state automatically the next time `step()` runs.

## What the Base `Optimizer` Does

The abstract base class `sadl.Optimizer` is responsible for:

- storing the named parameters to optimize (as an `OrderedDict`)
- storing optimizer state tensors (global and per-parameter)
- performing backpropagation with `backward(loss)`
- clearing gradients with `zero_grad(...)`
- moving optimizer state with `copy_to_device(...)`
- saving and loading optimizer state

The actual update rule itself is defined by `step()`, which each concrete optimizer implements.

## State

Some optimizers need additional tensors besides the parameters themselves.
Typical examples are momentum vectors or running variance estimates.

In `SADL`, these tensors form the optimizer state.

The state is stored in a single `OrderedDict` (`_state`) with two levels:

- **Global entries** like `lr`, `beta_1`, `t`, etc. sit at the top level.
- **Per-parameter entries** live under the `params` key, keyed by the parameter's name in the model state dict (e.g. `layers[0].W`).

For example, the internal state of an Adam optimizer might look like:

```python
optimizer._state = OrderedDict({
    "lr": tensor(0.001),
    "beta_1": tensor(0.9),
    "beta_2": tensor(0.999),
    "epsilon": tensor(1e-8),
    "t": tensor(3),
    "weight_decay": tensor(0.01),
    "params": OrderedDict({
        "layers[0].W": OrderedDict({"m": tensor(...), "v": tensor(...)}),
        "layers[0].b": OrderedDict({"m": tensor(...), "v": tensor(...)}),
        "layers[1].W": OrderedDict({"m": tensor(...), "v": tensor(...)}),
        "layers[1].b": OrderedDict({"m": tensor(...), "v": tensor(...)}),
    }),
})
```

Per-parameter state is created lazily by `_get_or_create_param_state`, which calls the subclass hook `_init_param_state` the first time a parameter is encountered during `step()`.

Important:
the optimizer state is **not** the same thing as the parameters.
For example, `copy_to_device(...)` on an optimizer moves the optimizer state, not the model parameters themselves.
Moving the model parameters is the job of `model.copy_to_device(...)`.

That is why training code often does both:

```python
model = model.copy_to_device(device)
optimizer = optimizer.copy_to_device(device)
```

Even though for vanilla SGD, the optimizer state may be empty.

### Why string keys?

The optimizer receives parameters as an `OrderedDict[str, Parameter]` from `model.get_parameters()`, and uses the same string names (like `"layers[0].W"`) to key both the parameters and their associated state.

This is a design choice. An alternative would be to key by the `Parameter` object itself (as PyTorch does at runtime), which avoids coupling the optimizer to the model's naming scheme. However, in a framework where transparency is a core goal, string keys have clear advantages:

- **Human-readable**: inspecting `optimizer._state["params"]` immediately shows which parameter each state entry belongs to.
- **Naturally serializable**: saving and loading state requires no mapping layer —> the same string keys work on disk and at runtime.
- **Consistent with `Function`**: the same names returned by `model.get_parameters()` appear in the optimizer state, so the two always agree.

The tradeoff is that renaming model attributes invalidates saved optimizer state. In practice, model attribute names change far less often than the optimizer state gets saved and loaded.

## Built-In Optimizers

`SADL` currently provides:

- `sadl.SGD`
- `sadl.Adam`

`SGD` can also act as momentum SGD through `friction`, and `Adam` can act as `AdamW` by setting `weight_decay > 0`.

## Properties

- `lr`: The learning rate (get/set).
- `state`: View of all tensors belonging to the optimizer state.
- `device`: Tuple of devices on which the optimizer state currently lives.
- `update_params`: Iterator over `(name, param, grad)` pairs that are trainable and have a gradient.

## Methods

- `backward(loss)`: Perform backpropagation with respect to a scalar loss tensor.
- `step()`: Update the tracked parameters using their current gradients.
- `zero_grad(additional_tensors=None)`: Clear gradients of the tracked parameters.
- `get_state(to_device=None)`: Return the optimizer state as an `OrderedDict`.
- `load_state(state, match_device, partial)`: Load optimizer state from an `OrderedDict`.
- `copy_to_device(device)`: Move optimizer state to a target device.

## Creating Custom Optimizers

To create a custom optimizer, subclass `Optimizer` and implement `step()`.

If the optimizer needs per-parameter state (like momentum or running averages), override `_init_param_state` and use `_get_or_create_param_state` in `step()`:

```python
import sadl
from sadl.tensor import zeros_like

class MyOptimizer(sadl.Optimizer):
    def __init__(self, params, *, lr=1e-3):
        super().__init__(params=params, lr=lr)

    def _init_param_state(self, param):
        return OrderedDict({
            "velocity": zeros_like(param, dtype=param.dtype, requires_grad=False),
        })

    @sadl.no_grad_fn
    def step(self):
        lr = self.lr
        for name, param, grad in self.update_params:
            s = self._get_or_create_param_state(name, param)
            s["velocity"] = 0.9 * s["velocity"] + grad
            param[...] = param - lr * s["velocity"]
```

The base class handles everything else: state traversal, serialization, device management, and gradient clearing.

## Summary

If I had to compress the role of the optimizer in `SADL` into one sentence, I would say:

The optimizer is the bridge between the computation graph and parameter updates, but that bridge only works if the parameters it tracks actually lie on the graph rooted at the loss.

Further reading:
- How the graph is built -> [`autograd/COMPUTATION_GRAPH.md`](autograd/COMPUTATION_GRAPH.md)
- How backpropagation walks the graph -> [`autograd/BACKPROPAGATION.md`](autograd/BACKPROPAGATION.md)
- Where operation-specific gradients come from -> [`autograd/GRADIENTS.md`](autograd/GRADIENTS.md)
