# Computation Graph


## How Autodiff Works

SADL implements reverse-mode automatic differentiation (backpropagation) using a dynamic computation graph, similar to PyTorch.

### The Computation Graph

In SADL, **Tensors are the computation graph**. There is no separate graph data structure. Each Tensor stores a `src` attribute pointing to the Tensors it was created from. This forms a back-referencing graph where each node knows its parents, but parents do not know their children:

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

This is intentional. Deep learning frameworks optimize for backward traversal because that is what backpropagation requires. Starting from the loss, we follow `src` references backward through the graph to compute gradients. Forward references (parent to child) are unnecessary and would only consume memory.

### Forward Pass: Building the Graph

When operations are performed on Tensors with `requires_grad=True`, the graph builds itself automatically:

1. The `Tensor` class overrides `__array_ufunc__` and `__array_function__` to intercept NumPy operations
2. Each operation creates a new Tensor that stores:
  - `src`: References to input tensors (the parents in the graph)
  - `backward_fn`: The gradient function for this operation
  - `op_ctx`: Any context needed for gradient computation (axis, masks, etc.)
3. The graph grows dynamically as operations execute

```python
x = sadl.tensor([1.0, 2.0], requires_grad=True)  # leaf, src = ()
y = sadl.tensor([3.0, 4.0], requires_grad=True)  # leaf, src = ()
z = x * y        # z.src = (x, y), z.backward_fn = mul_backward
loss = np.sum(z) # loss.src = (z,), loss.backward_fn = sum_backward
```

A more complex example:

```
a = tensor(...)      # leaf
b = tensor(...)      # leaf
c = tensor(...)      # leaf

d = a + b            # d.src = (a, b)
e = d * c            # e.src = (d, c)
f = np.sum(e)        # f.src = (e,)

Graph (following src backwards from f):

    f
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

### Backward Pass: Computing Gradients

When `optimizer.backward(loss)` is called:

1. **Topological Sort**: The graph is traversed from the loss tensor to find all nodes, ordered so that each node appears after all nodes that depend on it. This uses an iterative stack-based algorithm to avoid recursion limits on deep graphs.
2. **Gradient Propagation**: Starting from the loss (seeded with gradient 1.0), each node's `backward_fn` is called with:
  - The input tensors (`src`)
  - Which inputs need gradients (`compute_grad`)
  - The upstream gradient (`grad_out`)
  - Operation context (`op_ctx`)
3. **Gradient Accumulation**: Gradients flow backward through the graph. When a tensor is used in multiple operations, gradients are summed.
4. **Graph Cleanup**: After backpropagation, the graph structure is cleared to free memory. Parameter gradients are retained for the optimizer step.

### Gradient Operations Registry

Each supported operation has a corresponding backward function registered in `grad_ops.py` with metadata (op type inspired by [tinygrad](https://github.com/tinygrad/tinygrad)):

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

The `@broadcastable` decorator handles gradient reduction when inputs were broadcast during the forward pass.

### Supported Operations

Unary: `abs`, `negative`, `sqrt`, `square`, `exp`, `log`, `sin`, `cos`

Binary: `add`, `subtract`, `multiply`, `divide`, `power`, `matmul`, `maximum`, `minimum`

Reductions: `sum`, `mean`, `max`, `min`
