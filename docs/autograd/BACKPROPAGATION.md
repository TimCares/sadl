# Backpropagation

Once the computation graph has been built during the forward pass, backpropagation is the process of walking that graph backwards and accumulating gradients.

In `SADL`, this logic lives in `Optimizer.backward(...)`.

## The High-Level Idea

Given a scalar loss tensor, `SADL` does four main things:

1. Topologically order the graph.
2. Seed the loss with gradient `1`.
3. Traverse the nodes in reverse topological order and call each node's `backward_fn`.
4. Clear graph structure afterwards to free memory.

That is the whole algorithm at a high level.

## Seeding the Loss Gradient

The loss is the end of the computation.
So its gradient with respect to itself is `1`.

That is why `Optimizer.backward(...)` starts with:

```python
loss.grad = ones_like(loss).data  # "ones_like(loss)" creates a Tensor, but we need the underlying NDArray
```

This is the seed from which all other gradients are derived.

## The Main Backward Loop

The core loop in `Optimizer.backward(...)` looks like this in spirit:

```python
for node in reversed(node_order):
    if node.grad is None or node.is_leaf():
        continue

    src_requires_grad = [t.requires_grad for t in node.src]
    if not any(src_requires_grad):
        continue

    src_grads = node.backward_fn(
        *[s.data for s in node.src],
        compute_grad=src_requires_grad,
        grad_out=node.grad,
        **node.op_ctx,
    )

    for src, src_grad in zip(node.src, src_grads, strict=True):
        if src_grad is None:
            continue
        src.grad = src_grad if src.grad is None else src.grad + src_grad
```

You can think of one backward step like this:

```
Current node:

        node
      grad_out
         │
         ▼
      backward_fn
       ╱      ╲
      ▼        ▼
   grad_x    grad_y
     │          │
     ▼          ▼
     x          y
```

The current node already has its upstream gradient `grad_out`.
Its `backward_fn` turns that into gradients for the parent tensors (`src`), and those are then accumulated into `x.grad`, `y.grad`, and so on.

This is the key pattern:

- each node (Tensor) receives an upstream gradient `grad_out`
- its `backward_fn` computes gradients for its parents
- those gradients are accumulated into the parent tensors

## Why Gradients Are Accumulated

A tensor can contribute to the loss through multiple downstream paths.
In that case, multiple gradient contributions arrive at the same node.

So gradients are not overwritten.
They are summed:

```python
src.grad = src_grad if src.grad is None else src.grad + src_grad
```

That is exactly the multivariable chain rule at work.

This is also why we need topological order:
we want a node to be processed only after all later nodes have already contributed to its gradient.
See the optional note on `toposort(...)` at the end of this page.

## Cleanup

After the backward pass, `SADL` clears graph structure from the traversed nodes:

- `src`
- `backward_fn`
- `op_ctx`

and, for non-persistent activations, also `grad`.

This is done so the graph does not unnecessarily remain in memory after the backward pass has finished.
In Python, objects remain alive as long as something still references them.
By clearing `src`, many intermediate result tensors lose the references that keep entire graph fragments connected.
That makes it possible for Python's garbage collector to reclaim that memory later.

Parameters are the main exception, because their gradients need to survive long enough for the optimizer step.

That is what the attribute `keep_grad` in `Tensor` is for.
Since `Parameter` is a subtype of `Tensor`, it also has `keep_grad`, and `Parameter` sets it to `True` by default.
For regular tensors, `keep_grad` defaults to `False`, which is usually exactly what you want.

Setting `keep_grad=True` on intermediate tensors is possible, but should be done deliberately.
It means their gradients survive the cleanup step, which is useful for inspection or debugging, but also keeps more state alive than usual.

## Where the Gradient Formulas Come From

`Optimizer.backward(...)` does not itself know how to differentiate multiplication, matrix multiplication, `sum`, `mean`, or anything else.

It only knows how to:

- traverse the graph
- call the correct `backward_fn`
- accumulate the returned gradients

The actual local derivative rules come from the gradient registry.

That is the topic of the next doc:
[`GRADIENTS.md`](GRADIENTS.md)


## Optional: Why `toposort(...)` Is Needed

Before gradients can flow backwards, the graph has to be ordered so that each node is processed only after all nodes depending on it have already pushed gradients into it.

That is what `toposort(loss)` in `sadl/optimizer.py` does.

Conceptually:

```python
node_order = toposort(loss)
for node in reversed(node_order):
    ...
```

The first list, `node_order`, goes from leaves to loss.
The backward loop then reverses it, so it runs from loss back to the leaves.

For example:

```
Forward graph:

    loss
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

A valid topological order would be:

```python
[a, b, d, c, e, loss]
```

So the backward pass iterates in reverse:

```python
[loss, e, c, d, b, a]
```

The important part is not the exact order among siblings.
The important part is only that a node is processed after everything downstream of it has already had a chance to contribute to its gradient.
This is because one node can contribute to multiple later results.
For example:

```python
import numpy as np
import sadl

x = sadl.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = sadl.Parameter(sadl.tensor([1.0, 1.0, 1.0]))

a = x * w
b = a + 1
c = a * 2
loss = np.sum(b + c)
```

This creates a graph like:

```
      loss
        │
        ▼
     (b + c)
      ╱   ╲
     ▼     ▼
     b     c
      ╲   ╱
        ▼
        a
       ╱ ╲
      ▼   ▼
      x   w
```

Here, `a` contributes to the loss through **two** paths:

- `a -> b -> loss`
- `a -> c -> loss`

So before `a` is allowed to pass gradients further back to `x` and `w`, it first has to receive both contributions, from the `b` side and from the `c` side.

That is why the exact sibling order of `b` and `c` does not matter, but both of them must be processed before `a`.
Otherwise, `a.grad` would still be incomplete when `a.backward_fn(...)` is called, and `x` and `w` would receive gradients that are missing one path, and therefore
once indirect contribution to the loss.
