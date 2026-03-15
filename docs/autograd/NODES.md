# Nodes

In `SADL`, tensors do not just hold data.
Once autodiff enters the picture, they also become the nodes of the computation graph.

That is the key step from "a thin wrapper around array data" to "something that can participate in training."

Every time an operation on tensors is executed, `SADL` creates a new result tensor.
If gradient tracking is active, that result tensor can also store enough information to later answer questions like:

- Which tensors was this value created from?
- Which backward function belongs to this operation?
- Which extra context from the forward pass is needed during backpropagation?

This is why in `SADL`, the computation graph is not a separate giant data structure living somewhere else.
The tensors themselves are the graph nodes.

## What a Tensor Stores for Autodiff

Besides the actual array data, a `Tensor` in `SADL` has a few attributes that matter for autodiff:

Gradient-related:
- `grad`: the gradient accumulated for this tensor
- `requires_grad`: whether this tensor should participate in gradient tracking
- `keep_grad`: whether its gradient should be kept after backpropagation

Backprop-related:
- `src`: the parent tensors from which this tensor was created
- `backward_fn`: the backward function for the operation that created it
- `op_ctx`: extra context from the forward pass needed later in the backward pass

So the jump from "plain array wrapper" to "graph node" is actually quite small.
A tensor becomes a node simply by remembering where it came from and how gradients should flow through it.

## Leaves and Intermediate Nodes

Not every tensor plays the same role in the graph.

Leaf tensors are the starting points.
Typical examples are input tensors and parameters.
They do not have parents:

```python
x = sadl.tensor([1.0, 2.0], requires_grad=True)
x.src  # ()
```

Intermediate tensors are created by operations and therefore do have parents:

```python
x = sadl.tensor([1.0, 2.0], requires_grad=True)
y = sadl.tensor([3.0, 4.0], requires_grad=True)
z = x * y

z.src  # (x, y)
```

So in `SADL`, each new operation can extend the graph simply by returning a new tensor whose `src` points to earlier tensors.

There is no separate "graph node object" that wraps a tensor, and no second structure that mirrors the values somewhere else.
The value and the node are the same object.

That keeps the implementation small and readable:

- values live in tensors
- graph structure also lives in tensors
- backpropagation can just start from the loss tensor and walk through `src`

This naturally leads to two next questions:

1. How exactly is that graph [represented](COMPUTATION_GRAPH.md)?
2. How is it traversed during [backpropagation](BACKPROPAGATION.md)?

That is what the next docs are about.
