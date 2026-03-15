# Computation Graph

`SADL` uses a dynamic computation graph _by default_.
That means the graph is built during the forward pass, exactly while the user code is running.

## The Main Idea

In `SADL`, **tensors are the computation graph**.
There is no separate graph object living somewhere else.
Each result tensor stores references to the tensors it was created from.

This forms a back-referencing graph where each node knows its parents, but parents do not know their children:

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

This is intentional.
Backpropagation starts at the loss and walks backwards.
So storing parent references is enough, while child references would only consume extra memory.

## Where the Graph Is Built

The graph is built through the combination of:

1. `Tensor.__array_ufunc__`
2. `Tensor.__array_function__`
3. `sadl.dispatch.dispatch_op(...)`

The tensor methods intercept NumPy-style operations.
Then `dispatch_op(...)` performs the real work.

This is why `sadl/dispatch.py` is such an important file:
it is the place where a normal numerical operation turns into a tracked graph operation.

## What `dispatch_op(...)` Does

The logic in `dispatch_op(...)` is compact, but it does several important things in one place:

1. Unwrap `Tensor` inputs to their underlying array data.
2. Determine the device on which the op should run.
3. Select the correct backend, NumPy or CuPy.
4. Execute the forward operation in the array backend.
5. Wrap the result back into a `Tensor`.
6. If gradients matter, attach graph metadata to the result tensor.

So the _dynamic_ computation graph is not built by a separate "graph builder".
It is built exactly at the moment the forward operation executes.

## A Simplified View of the Implementation

This is not the full implementation, but it matches the real structure of `sadl/dispatch.py`:

```python
def dispatch_op(op_name, *, op_fn, op_inputs, **kwargs):
    input_args = [a.data if isinstance(a, Tensor) else a for a in op_inputs]

    device = _determine_and_ensure_device(input_args)
    backend = get_array_module_from_device(device)

    result = op_fn(*input_args, **kwargs)
    result = backend.asarray(result)

    if not grad_tracking:
        return Tensor(result, requires_grad=False)

    src = tuple(_to_tensor(i, device=device) for i in op_inputs)
    result_requires_grad = any(elem.requires_grad for elem in src)
    backward_fn = get_grad_op(op_name)

    result_tensor = Tensor(result, requires_grad=result_requires_grad)

    if result_requires_grad:
        result_tensor.src = src
        result_tensor.backward_fn = backward_fn
        result_tensor.op_ctx = kwargs

    return result_tensor
```

That is really the heart of how the graph is built in `SADL`.

## Why This Place Is So Natural

I think this is a particularly clean design, because at the exact moment an operation runs, `SADL` already knows everything it needs:

- which operation is being executed
- which inputs participated
- which backend is active
- which extra context should be remembered

So `dispatch.py` is the natural place where graph nodes are born.

## What Gets Stored in a Node

If the result of an operation requires gradients, `dispatch_op(...)` stores three important things on the resulting tensor:

- `src`: the parent tensors
- `backward_fn`: the backward function for this operation
- `op_ctx`: extra context from the forward pass, for example `axis`, `keepdims`, or masks

This is why graph building stays local and cheap.
Each node only stores what it personally needs for the backward pass later.

There is also one small but important detail:
even non-`Tensor` inputs can be converted into tensors for the graph via `_to_tensor(...)`.
That keeps the graph structurally consistent: There are only Tensors.

## Special Cases

Some operations need extra information from the forward pass.

For example, `max`, `min`, `maximum`, and `minimum` create masks during the forward pass and store them in `op_ctx`.
These masks are then used later during backpropagation to decide where gradients should flow.
This is not strictly necessary, as these masks can also be computed during the backward pass, but it is a nice little optimization.

So `dispatch.py` does not just connect nodes.
It also captures forward-time information that later becomes essential in the backward pass.

## A Small Example

```python
import numpy as np
import sadl

x = sadl.tensor([1.0, 2.0], requires_grad=True)
y = sadl.tensor([3.0, 4.0], requires_grad=True)

z = x * y
loss = np.sum(z)
```

After these operations:

- `x` and `y` are leaves, so `x.src == ()` and `y.src == ()`
- `z` was created by multiplication, so `z.src == (x, y)`
- `loss` was created by reduction, so `loss.src == (z,)`

We can sketch that as:

```
loss
 │
 ▼
 z
╱ ╲
▼  ▼
x  y
```

The forward pass just tracks operations and creates this graph one operation at a time.
Nothing more than that happens.

Further reading:
- How the graph is traversed backwards -> [`BACKPROPAGATION.md`](BACKPROPAGATION.md)
- Where `backward_fn` values come from -> [`GRADIENTS.md`](GRADIENTS.md)
