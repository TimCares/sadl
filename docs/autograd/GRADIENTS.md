# Gradients

When `Optimizer.backward(...)` walks the graph, it needs to know how to differentiate each individual operation.

For example:

- if `z = x * y`, what are the gradients with respect to `x` and `y`?
- if `z = np.sum(x)`, how do we expand the upstream gradient back to the shape of `x`?
- if broadcasting happened during the forward pass, how do we reduce the gradient correctly?

Those local derivative rules live in `sadl/grad_ops.py`.

## The Gradient Registry

`SADL` keeps a registry that maps forward operation names to backward functions.

That means when `dispatch_op(...)` sees an operation like `multiply` or `sum`, it can later attach the correct `backward_fn` to the resulting tensor.

The central API looks like this:

| Export | Description |
|--------|-------------|
| `get_grad_op(name)` | Get the backward function by forward op name |
| `register_grad_op(...)` | Register a new backward function |
| `get_grad_op_spec(name)` | Get the full metadata for an op |
| `OpType` | Operation category like `ELEMENTWISE`, `REDUCTION`, `MOVEMENT`, `LINALG` |
| `OpInputs` | Expected number of tensor inputs |
| `GradOpSpec` | Dataclass bundling backward function and metadata |

The last 4 are mainly used for testing, so `get_grad_op(name)` and `register_grad_op(...)` are the important ones.

## Registration

Backward functions are registered with a decorator.

For example, multiplication is registered like this:

```python
@register_grad_op(
    op_type=OpType.ELEMENTWISE,
    op_inputs=OpInputs.BINARY,
    forward_names=("mul", "multiply"),
)
@broadcastable
def mul_backward(*inputs, compute_grad, grad_out):
    x, y = inputs
    grad_x = y * grad_out if compute_grad[0] else None
    grad_y = x * grad_out if compute_grad[1] else None
    return grad_x, grad_y
```

The important thing here is that the backward function computes **local gradients** for the direct parents of the current node.
It does not traverse the graph itself.
That job belongs to `Optimizer.backward(...)`.

## What a Backward Function Receives

A backward function typically receives:

- the forward inputs
- `compute_grad`: flags indicating for which parents gradients are actually needed
- `grad_out`: the upstream gradient arriving from later in the graph
- additional context from `op_ctx`

So conceptually, a backward function answers:

"Given the gradient of the output, what are the gradients of the inputs?"

## Reductions and Context

Some operations need extra information from the forward pass.

For example, `sum` needs to know things like `axis` and `keepdims` in order to expand the upstream gradient back to the original input shape:

```python
@register_grad_op(
    op_type=OpType.REDUCTION,
    op_inputs=OpInputs.UNARY,
)
def sum_backward(*inputs, compute_grad, grad_out, **kwargs):
    x = inputs[0]
    axis = make_axis(ndim=x.ndim, kwargs_dict=kwargs)

    if not kwargs.get("keepdims", False):
        grad_out = np.expand_dims(grad_out, axis=axis)

    grad_x = np.broadcast_to(grad_out, shape=x.shape)
    return (grad_x,)
```

That is exactly why `dispatch.py` stores `op_ctx` on the result tensor during the forward pass.
The backward function may need that context later.

## Broadcasting

Broadcasting is one of the main reasons gradient code can become annoying.

If an op broadcast one input during the forward pass, the backward pass has to reduce the gradient back to the original input shape.

That is why several backward functions in `SADL` use the `@broadcastable` decorator.
It handles that bookkeeping so the local gradient rule itself can stay readable.

## Supported Operation Families

Right now the registry covers several important families of operations:

- unary ops like `exp`, `log`, `sin`, `cos`, `sqrt`
- binary ops like `add`, `subtract`, `multiply`, `divide`, `power`
- linear algebra like `matmul`
- reductions like `sum`, `mean`, `max`, `min`
- movement / utility ops like `copy_to_device`, `reshape`, `astype`

## The Division of Responsibility

I think it helps to keep the responsibilities separate in your head:

- `dispatch.py` builds graph nodes and attaches the right `backward_fn`
- `grad_ops.py` defines how each individual operation is differentiated
- `Optimizer.backward(...)` traverses the graph and calls those functions in the correct order

That separation is one of the reasons the autodiff code in `SADL` stays relatively readable.
