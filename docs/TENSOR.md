# Tensor

## TL;DR

If I had to compress the whole idea of `Tensor` into one sentence, it would be this:

`Tensor` is a thin object that stores array data, forwards operations to the correct backend, and adds just enough structure for `SADL` semantics.

Without talking about autodiff yet, these semantics are mainly:

- device awareness
- a consistent user-facing type
- dispatch through NumPy protocols
- a few utility methods that are genuinely convenient

That is why `Tensor` in `SADL` is intentionally not a huge abstraction.
It is a thin but important layer.

## Wrapping `NDArray`

The [`Tensor`](../sadl/tensor.py) class is the central data-holding class in `SADL`.
This is no different from e.g. PyTorch or tinygrad.

`Tensor` works as a wrapper around `NDArray` (a type alias of `np.ndarray`), which is defined by the backend and is used as the shared array type in `SADL`.
`NDArray` can be used as a static type, while at runtime the actual array may be either a NumPy array or a CuPy array.

The array is stored in the attribute `data` of the `Tensor` class, and all operations performed on `Tensor` are actually performed on this `data`.

```python
from numpy.lib.mixins import NDArrayOperatorsMixin
from sadl.backend import NDArray


class Tensor(NDArrayOperatorsMixin):
    def __init__(
        self,
        data: NDArray,
        *,
        requires_grad: bool = False,
        keep_grad: bool = False,
    ) -> None:
        self.data = data  # NumPy or CuPy array
        ...
```

This is simplified, the actual implementation contains more, but it shows the main idea: a `Tensor` stores array data and builds additional semantics around it.

Omitting the gradient part for now, `Tensor` just receives an `NDArray`, NumPy or CuPy, and stores it as an attribute.
But how do operations on `Tensor` work then? The answer is: they do not really work on `Tensor` itself, but on `data`.

This is where protocols like `__array_ufunc__` and `__array_function__` come in (as introduced in [BACKEND.md](BACKEND.md)).
As you might have noticed, `Tensor` inherits from `NDArrayOperatorsMixin`,
which is basically a way to tell NumPy that `Tensor` implements operator behavior through `__array_ufunc__`.

`Tensor` now implements both `__array_ufunc__` and `__array_function__`:

```python
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from sadl.backend import NDArray

class Tensor(NDArrayOperatorsMixin):

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        op_name = normalize_grad_op_name(name=ufunc.__name__, is_reduce=method == "reduce")

        func = getattr(ufunc, method)

        from .dispatch import (  # noqa: PLC0415 (avoid circual import between tensor.py and dispatch.py)
            dispatch_op,
        )

        return dispatch_op(op_name, op_fn=func, op_inputs=inputs, **kwargs)

    def __array_function__(
        self,
        func: Any,
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        op_name = normalize_grad_op_name(name=func.__name__)

        from .dispatch import (  # noqa: PLC0415 (avoid circual import between tensor.py and dispatch.py)
            dispatch_op,
        )

        return dispatch_op(op_name, op_fn=func, op_inputs=args, **kwargs)
```

What this means is: any NumPy op in which `Tensor` takes part is not executed by NumPy directly, but instead routed through one of these protocol functions.
So we effectively catch operations like `my_tensor * 3`, `np.sum(my_tensor)`, `np.matmul(my_tensor, my_second_tensor)`, or even mixed cases like `np.matmul(my_tensor, my_numpy_array)`.

From there, we forward:

- the name of the operation
- the actual callable that performs it
- all inputs participating in the operation

to a dedicated `dispatch_op` function.

That function is responsible for selecting the right backend, NumPy or CuPy, executing the operation, and later also attaching autodiff metadata.
For now, the important part is simply this: the result is again returned as a `Tensor`, so it can continue participating in further operations.

This has a huge advantage: we can support a large amount of functionality without writing a manual wrapper for every single operation.
The client can mostly just use NumPy, but gets `SADL` `Tensor`s.


## Members

`Tensor` exposes some properties and methods that are frequently useful when working with NumPy/CuPy arrays.
Some of these are just thin wrappers around `data`, while others enforce the correct `SADL` semantics, especially around devices.

**Properties (all thin array wrappers)**:
- `device`: The current device the `Tensor` is on. It is of type `sadl.TensorDevice`.
- `shape`: Shape of the `Tensor`.
- `ndim`: Number of dimensions the Tensor has.
- `size`: Number of elements in the Tensor.
- `dtype`: The datatype of the `Tensor`, represented as a NumPy dtype.

**Methods:**
Array wrappers:
- `astype(np.dtype)`: Create a copy of the `Tensor` with a different NumPy dtype.
- `item()`: If the Tensor is a scalar value, get the actual scalar value, e.g. float.

Device handling and graph-related utilities:
- `copy_to_device(device)`: Copy the `Tensor` to CPU or any available GPU device.
- `detach()`: Detach from the current computation graph.
- `cpu()`: Move to CPU.
- `gpu(device_id=0)`: Move to GPU with device id 0.

The important thing here is that `Tensor` still stays close to the underlying array.
It does not try to replace the full NumPy API with a second, parallel `Tensor` API.

## Creating Tensors

A `Tensor` can be created in three ways:

1. Directly via `Tensor.__init__`.
A `Tensor` can be created by passing a NumPy or CuPy array to the `Tensor` constructor.

```python
import numpy as np
import sadl

arr = np.array([1, 2, 3])

tensor = sadl.Tensor(arr)
```

However, this is usually not the encouraged way, because you cannot pass in scalar values, lists, or general array-like data directly.
`Tensor.__init__` simply accepts already-created array data.
So in practice, it is mostly useful internally, or when you already explicitly have a NumPy or CuPy array.

2. `tensor` factory function

Similar to PyTorch, this is the encouraged way to create a new `Tensor`.

```python
import numpy as np
import sadl

cpu = sadl.TensorDevice("cpu")
gpu = sadl.TensorDevice("cuda", device_id=0)

tensor1 = sadl.tensor(1)
tensor2 = sadl.tensor([1, 2, 3], dtype=np.float32)
tensor3 = sadl.tensor([[1, 2], [3, 4]], device=gpu, requires_grad=True)
tensor4 = sadl.tensor(tensor3, device=cpu)  # creates a new copy on cpu
```

This function is convenient because it accepts general input data, handles dtype conversion, and allows the device to be chosen explicitly.

3. Array creation routines

This part is easy to overlook.

Even though the library used to perform most ops in `SADL` is just NumPy, meaning clients will use operations like `np.sum`,
you cannot use every NumPy array creation primitive, like `np.zeros_like` or `np.eye`, and expect the result to automatically be a `Tensor`.

The reason is that array creation does not start from an existing `Tensor`, so there is nothing that would dispatch the operation back into `SADL`.

There are some options to use these creation routines:

1. Use the built-in creation helpers provided by `SADL`, namely `ones`, `ones_like`, `zeros`, `zeros_like`, and `eye`.

Again, I did not provide a huge zoo of built-in creation routines, because these tend to be the most frequently used ones and I do not want to bloat the codebase.
That would go against the whole philosophy.

```python
import sadl

x = sadl.ones((2, 3))
y = sadl.zeros_like(x)
I = sadl.eye(4)
```

2. Numpy

You can use NumPy creation routines, but then you have to convert the result into a `Tensor`.
For that, it is encouraged to use the `tensor` factory function.

```python
import sadl
import numpy as np

array = np.identity(3)

tensor = sadl.tensor(array)
```

If you want the `Tensor` on CUDA, you can do the following:

```python
import sadl
import numpy as np

array = np.identity(3)
cuda = sadl.TensorDevice("cuda")

tensor = sadl.tensor(array, device=cuda)
```

Or, even better, if you want to avoid an unnecessary CPU-to-GPU round-trip:

```python
import sadl
from sadl.backend import get_array_module_from_device

cuda = sadl.TensorDevice("cuda")
xp = get_array_module_from_device(cuda)

array = xp.identity(3)

tensor = sadl.tensor(array, device=cuda)
```

This way the array is created directly on the target backend.

## Limitations

For most operations, you use `np.<op>` instead of `Tensor.<op>`.
Typical exceptions are utility methods like `Tensor.astype(...)`, `Tensor.copy_to_device(...)`, `Tensor.cpu()`, or `Tensor.gpu()`.

Why?
Because otherwise I would have to write a wrapper for each operation, which would look something like this:

```python
import numpy as np

class Tensor:
    def sum(self, axis):
        return np.sum(self, axis=axis)

    def argmax(self, axis):
        return np.argmax(self, axis=axis)

    ...
```

This is simplified, but it shows the point.

It would add a lot of boilerplate, bloat the codebase, and not really teach us anything new.
So `SADL` keeps the NumPy-style API for most operations on purpose.


## Further Reading

To continue the story: "How tensors are turned into autodiff nodes" -> [`autograd/NODES.md`](autograd/NODES.md).

Why device handling is more complicated than it seems -> [`DEVICE.md`](DEVICE.md).

How `SADL` serializes tensors -> [`DISK.md`](DISK.md).
