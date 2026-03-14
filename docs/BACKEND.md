# Backend

If `SADL` wants to stay small and readable, then one question naturally follows:
How much of the heavy lifting can we reuse instead of rebuilding it?

For linear algebra, arrays, dtypes, and most numerical operations, the answer is: a lot.

## Reusing NumPy Without Any Tensor Yet

Before even talking about `Tensor`, there is already a very important idea:
you can get surprisingly far by just building on top of the NumPy API.

NumPy already gives us:

- arrays and memory buffers
- element-wise operations
- matrix multiplication and linear algebra
- dtypes
- a familiar API that most Python users already know

So instead of inventing a completely different world, `SADL` tries to stay as close as possible to this one.

## Why the NumPy API Is So Valuable

One really nice thing about the scientific Python ecosystem is that the API itself is often more important than the concrete implementation underneath.

That is exactly where protocols like `__array_ufunc__` and `__array_function__` become interesting.

In short:

- `__array_ufunc__` lets an array type intercept NumPy ufuncs like `np.add`, `np.multiply`, or `np.exp`
- `__array_function__` lets an array type intercept higher-level NumPy functions like `np.sum`, `np.mean`, or `np.matmul`

This means that if another array library implements these hooks, NumPy code can still work, even if the actual array is not a plain `np.ndarray`.

```python
import numpy as np

class MyArray:
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        print(f"Intercepted ufunc: {ufunc.__name__}")
        return "handled"

x = MyArray()

print(np.exp(x))
# Intercepted ufunc: exp
# handled
```

That idea is powerful on its own.
It means NumPy is not just a concrete library, but also a kind of language many array libraries can speak.

## GPUs

Of course, deep learning cannot live on CPU alone forever.
From the point where models have tens of millions of parameters and datasets become similarly large, training on CPU becomes painfully slow.

Fortunately, most neural network operations, especially things like matrix multiplication, parallelize extremely well.
That is exactly what GPUs are good at.

Now comes the fun part: while NumPy itself does not run on CUDA, [CuPy](https://cupy.dev) does.

CuPy is an array library that can almost be used interchangeably with NumPy.
The API is extremely similar, but the operations run on the GPU.

```python
import cupy as xp
# or: import numpy as xp

x = xp.arange(6).reshape(2, 3).astype("f")

print(x)
# array([[ 0.,  1.,  2.],
#        [ 3.,  4.,  5.]], dtype=float32)

x_sum = x.sum(axis=1)

print(x_sum)
# array([  3.,  12.], dtype=float32)
```

This is huge, because now we do not need to rebuild the whole array universe ourselves just to get GPU support.

## Why NumPy and CuPy Fit So Well Together

And now the really beautiful part: CuPy is [interoperable](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#numpy) with NumPy.

Because CuPy implements NumPy's dispatch protocols, you can often use the NumPy API on CuPy arrays directly.
The call looks like NumPy, but the actual work is carried out by CuPy on the GPU.

```python
from typing import Any, Literal

import cupy as cp
import numpy as np

def create_array(data: list[Any], device: Literal["cpu", "cuda"]) -> np.ndarray:
    if device == "cpu":
        return np.array(data)
    return cp.array(data)

x = create_array([1, 2, 3], device="cuda")

y = np.sum(x)      # dispatched to CuPy
z = np.exp(x)      # also dispatched to CuPy

print(y)
print(z)
```

That is the key insight:
we can often program against the NumPy API, while deciding at runtime whether the actual array lives on the CPU or on the GPU.

## Turning This Into a Backend

Once you combine these ideas, a natural design starts to emerge:

1. Use NumPy semantics and NumPy dtypes everywhere.
2. Allow the concrete runtime array to be either `np.ndarray` or `cp.ndarray`.
3. Decide at runtime which array module, NumPy or CuPy, should execute an operation.

This is exactly the purpose of a backend layer.

The backend does not try to be clever.
It just answers questions like:

- What kind of array do I currently have?
- Am I on CPU or CUDA?
- Which array module should execute this operation?

In `SADL`, this is kept very small.
The idea is not to create yet another giant abstraction layer, but just enough indirection so the rest of the code can stay clean.

## A Small Example

This is the kind of logic the backend enables:

```python
import numpy as np

from sadl.backend import NDArray, TensorDevice, get_array_module_from_device

def sigmoid(x: NDArray, device: TensorDevice) -> NDArray:
    xp = get_array_module_from_device(device)
    return 1 / (1 + xp.exp(-x))

cpu = TensorDevice("cpu")
cuda = TensorDevice("cuda", device_id=0)

# same code path, different runtime backend
```

The important part is not the function itself.
The important part is that `xp` is selected dynamically, while the code using it stays basically identical.

This gives us a surprisingly clean setup:

- NumPy-style code
- NumPy dtypes
- one shared backend interface
- CPU or GPU decided at runtime

And because CuPy behaves so similarly to NumPy, the amount of extra code needed for this is refreshingly small.

## Static Types vs Runtime Reality

There is also a nice typing benefit here.

At runtime, arrays may be either NumPy arrays or CuPy arrays.
But for much of the code, we do not actually want that distinction to infect every function signature.

So the backend can expose a shared array type alias, `NDArray`, and a structural type for the array module itself.
That way, the rest of the code can mostly talk in one language, while the concrete backend is selected dynamically.

This is also why NumPy dtypes are used throughout the codebase.
They already describe exactly what we care about, and CuPy follows the same dtype model closely enough that we do not need a second dtype universe.

## Why I Like This Design

Personally, I think this is exactly the kind of reuse that is worth it.

It does not hide the core ideas.
It does not make the code magical.
And it saves us from reimplementing a huge amount of general-purpose array infrastructure that is already solved extremely well.

So instead of spending complexity on arrays, kernels, and dtype systems, `SADL` can spend it on the parts that are actually interesting here:
gradients, backpropagation, optimizers, and eventually `Tensor`.

That is the next [step](Tensor.md).
