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
optimizer = sadl.SGD(list(model.parameters), lr=0.01)
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
And when `step()` checks that parameter, it will fail because there is no gradient to use.

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
optimizer = sadl.SGD(list(model.parameters), lr=1e-2)

x = sadl.tensor([[1.0, 2.0, 3.0]])
loss = model(x).sum()

optimizer.backward(loss)
optimizer.step()  # fails: "unused.grad" is still None
```

Here `self.unused` is a real `Parameter`, so it is included in `model.parameters` and therefore tracked by the optimizer.
But it is never used in `__call__`, so it never appears in the graph rooted at `loss`.
That means backpropagation never reaches it.

This is why the forward pass and the optimizer parameter list must agree.

## Freezing or Excluding Parameters

The inverse is also true:
if a parameter should **not** be optimized, it should not be part of the optimizer parameter list.

In other words, the optimizer should only receive the parameters you really want to update.

This is especially important because `Optimizer.step()` expects its tracked parameters to have gradients.
So if you freeze parameters or want to exclude some of them, you should also exclude them from the optimizer.

## What the Base `Optimizer` Does

The abstract base class `sadl.Optimizer` is responsible for:

- storing the parameters to optimize
- storing optimizer state tensors
- performing backpropagation with `backward(loss)`
- clearing gradients with `zero_grad(...)`
- moving optimizer state with `copy_to_device(...)`
- saving and loading optimizer state

The actual update rule itself is defined by `step()`, which each concrete optimizer implements.

## State

Some optimizers need additional tensors besides the parameters themselves.
Typical examples are momentum vectors or running variance estimates.

In `SADL`, these tensors form the optimizer state.

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

## Built-In Optimizers

`SADL` currently provides:

- `sadl.SGD`
- `sadl.Adam`

`SGD` can also act as momentum SGD through `friction`, and `Adam` can act as `AdamW` by setting `weight_decay > 0`.

## Properties

- `state`: View of all tensors belonging to the optimizer state.
- `device`: Tuple of devices on which the optimizer state currently lives.

## Methods

- `backward(loss)`: Perform backpropagation with respect to a scalar loss tensor.
- `step()`: Update the tracked parameters using their current gradients.
- `zero_grad(additional_tensors=None)`: Clear gradients of the tracked parameters.
- `get_state(to_device=None)`: Return the optimizer state as an `OrderedDict`.
- `load_state(state, match_device, partial)`: Load optimizer state from an `OrderedDict`.
- `copy_to_device(device)`: Move optimizer state to a target device.

## Summary

If I had to compress the role of the optimizer in `SADL` into one sentence, I would say:

The optimizer is the bridge between the computation graph and parameter updates, but that bridge only works if the parameters it tracks actually lie on the graph rooted at the loss.

Further reading:
- How the graph is built -> [`autograd/COMPUTATION_GRAPH.md`](autograd/COMPUTATION_GRAPH.md)
- How backpropagation walks the graph -> [`autograd/BACKPROPAGATION.md`](autograd/BACKPROPAGATION.md)
- Where operation-specific gradients come from -> [`autograd/GRADIENTS.md`](autograd/GRADIENTS.md)
