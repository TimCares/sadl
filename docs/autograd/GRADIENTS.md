
The gradient for each `Tensor` (which is the gradient for `data`), is stored in the attribute `grad`.
This is also an `NDArray` and should have the same shape as `data`. If a Tensor has no gradient, `grad` will be `None`.

**Attributes:**
- `requires_grad`: Whether gradients are tracked
- `grad`: Stored gradient after backward pass
- `src`: Parent tensors in computation graph
- `keep_grad`: Whether to retain gradient after backward


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



## Creating Custom Backward Functions

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
