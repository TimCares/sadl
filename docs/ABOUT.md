# About
`SADL` builds on [`numpy`](https://numpy.org/doc/stable/index.html), one of the most popular scientific computing and array library in python.

Why? Because a Deep Learning framework consists of 3 major components:
1. Linear Algebra (think add, subtract, matrix multiplication, ...)
2. Gradient computation
3. Backpropagation

While 2. and 3. are more specific concepts somewhat exclusively used in Machine Learning, Linear Algebra is something more general: It appears everywhere and is one of the most well-know branches in maths.

This is an advantage, because most Computer Scientist are very familiar with it:
If you have used python, chances are pretty high that you have already used numpy or have at least heard of it.

Most Deep Learning frameworks, like Pytorch, Tensorflow, and TinyGrad, implement all three components from scratch.
This makes sense if you are being serious about building a framework used in enterprise-grade systems and for the
computationally most complex tasks, like training models with parameters in the trillions in giant datacenters.

For these cases, you need extreme efficiency, account for a lot of use-cases, and a lot of low-level utility and infrastructure code, with C-level kernels,
multi-node gradient communication, and distributed checkpointing just to name a few.

There is even quite the amusing story where engineers at Meta had to compute dummy operations to avoid stressing the power grid supplying the
data center when training LLaMa 3 (find the story [here](https://newsletter.semianalysis.com/p/ai-training-load-fluctuations-at-gigawatt-scale-risk-of-power-grid-blackout)).
While supporting these complex concepts is certainly impressive and of great value, it is exactly this what inflates the size of the codebases by several orders of magnitude,
making understanding how the core of these systems actually work under the hood incredible difficult.

Using numpy as the backend

**Note**: The goal of this framework is by no means to show some novel new idea, and I do not like labeling things as "novel" anyway.

Instead, the goal is to show what amazing things one can build on existing components with minimal and clear code.
And if you look at how the code works, you will find that the core of the algorithms powering the training of today's foundation models really aren't that difficult to understand.


# GPUs

GPUs are the bread and butter of every Deep Learning framework. From the point were the models you are training start to have tens of millions of parameters,
and your datasets have a similar magnitude, training on the CPU becomes painfully slow.
Fortunately, since most operations in Neural Networks, think Matrix Multiplication, can be parallelized, GPUs allow for massive speedup.

Since this speedup mainly affects components 1. and 2., the linear algebra backend needs to support computation on the GPU, e.g. via CUDA,
which numpy does not.

However, there is an often overlooked package which does exactly that: [cupy](https://cupy.dev).

Cupy is an array library which can almost be used interchangeably with numpy. The API is almost exactly the same, with one small difference: All operations run on the GPU!

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

This is huge, because (1) we do not need to change any code when we want to switch to using the GPU, and (2) it keeps the library laser-focused on what actually matters: **Gradients, backpropagation, and optimizers**.

And now the best part: Cupy is [interoperable](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#numpy) with numpy!

What does that mean? Cupy implements the numpy `__array_ufunc__` interface, meaning that if you call a numpy op on a cupy array,
instead of the numpy backend executing the operation on the cpu, which would throw an
error since cupy arrays only live on the gpu, the operation is dispatched to the cupy kernels.
Therefore, as long as the array creation happens via the cupy api, you can use most of the operations provided by the numpy package on cupy arrays.

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

x_sum = np.sum(x) # use numpy package, but run on gpu

print(x_sum)
```

This allows `SADL` Tensors to be _used with the numpy package/API regardless on which device the backing memory buffer of the Tensor_ (the arrays, so np.ndarray or cp.ndarray) _lives_.

The extra code necessary to support this, and abstract away if a numpy or cupy array is currently used, is minimal. The result: To the client there are just Tensors, and they
can live on the cpu or any gpu available, and can be moved between the devices. These are **exactly** the semantics provided by Pytorch and Tensorflow with almost **zero** overhead
and an API one couldn't be more familiar with!

Note: There are some nuances when moving Tensors between devices which other frameworks aren't quite transparent about, but `SADL` brings light into the dark: See more [here](./COPY_TO_DEVICE.md).

Also, there are some [differences between numpy and cupy](https://docs.cupy.dev/en/stable/user_guide/difference.html), so behavior might differ in edge cases depending if your Tensors are on the cpu or gpu.
As always: If you discover any unexpected behavior please let me know by opening an issue.
