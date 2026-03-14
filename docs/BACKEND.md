# Backend


## GPUs

GPUs are the bread and butter of every deep learning framework. From the point where the models you are training start to have tens of millions of parameters,
and your datasets have a similar magnitude, training on the CPU becomes painfully slow.
Fortunately, since most operations in neural networks, think matrix multiplication, can be parallelized, GPUs allow for massive speedups.

Since this speedup mainly affects components 1. and 2., the linear algebra backend needs to support computation on the GPU, e.g. via CUDA,
which numpy does not.

However, there is an often overlooked package that does exactly that: [CuPy](https://cupy.dev).

CuPy is an array library that can almost be used interchangeably with NumPy. The API is almost exactly the same, with one small difference: all operations run on the GPU.

```python
import cupy as xp
# or: import numpy as xp

x = xp.arange(6).reshape(2, 3).astype('f')

print(x)
# array([[ 0.,  1.,  2.],
#        [ 3.,  4.,  5.]], dtype=float32)

x_sum = x.sum(axis=1)

print(x_sum)
# array([  3.,  12.], dtype=float32)
```

This is huge, because (1) we do not need to change any code when we want to switch to using the GPU, and (2) it keeps the library laser-focused on what actually matters: gradients, backpropagation, and optimizers.

And now the best part: CuPy is [interoperable](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#numpy) with NumPy.

What does that mean? CuPy implements NumPy's `__array_ufunc__` interface, meaning that if you call a NumPy op on a CuPy array,
instead of the NumPy backend trying to execute the operation on the CPU, which would fail since CuPy arrays live on the GPU, the operation is dispatched to CuPy kernels.
Therefore, as long as the array creation happens via the CuPy API, you can use most of the operations provided by the NumPy package on CuPy arrays.

```python
from typing import Literal, Any
import cupy as cp
import numpy as np

def create_array(data: list[Any], device: Literal["cpu" | "cuda"]) -> np.ndarray:
    if device == "cpu":
        return np.array(data)
    # else, create on gpu:
    return cp.array(data)

x = create_array([1,2,3], device="cuda")

x_sum = np.sum(x)  # use the numpy API, but run on the GPU

print(x_sum)
```

This allows `SADL` tensors to be used with the NumPy package/API regardless of which device the backing memory buffer of the tensor, `np.ndarray` or `cp.ndarray`, lives on.

The extra code necessary to support this, and abstract away whether a NumPy or CuPy array is currently used, is minimal. The result: to the client there are just tensors, and they
can live on the CPU or any GPU available, and can be moved between devices. These are **exactly** the semantics provided by PyTorch and TensorFlow with almost **zero** overhead
and an API one couldn't be more familiar with!

Note: there are some nuances when moving tensors between devices that other frameworks are not always very transparent about, but `SADL` tries to make these trade-offs visible. See more [here](./COPY_TO_DEVICE.md).

Also, there are some [differences between NumPy and CuPy](https://docs.cupy.dev/en/stable/user_guide/difference.html), so behavior might differ in edge cases depending on whether your tensors are on the CPU or GPU.
As always, if you discover any unexpected behavior, please let me know by opening an issue.
