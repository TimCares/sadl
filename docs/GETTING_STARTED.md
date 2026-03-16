# Getting Started

A small guide to the main parts of `SADL`.

## Quick Start

`SADL` works similarly to PyTorch.

```python
import sadl

# Create tensors
gpu = sadl.TensorDevice("cuda", device_id=0)  # device_id defaults to 0
# cpu = sadl.TensorDevice("cpu")
x = sadl.tensor([[1.0, 2.0]], device=gpu)  # device defaults to cpu

# Build model
model = sadl.Mlp([
    sadl.Linear(dim_in=2, dim_out=4),
    sadl.ReLU(),
    sadl.Linear(dim_in=4, dim_out=1),
])
model = model.copy_to_device(gpu)

# Create an optimizer
optimizer = sadl.SGD(model.get_parameters(), lr=0.01)
optimizer = optimizer.copy_to_device(gpu)

# A single training step
output = model(x)
loss = output.sum()
optimizer.backward(loss)
optimizer.step()
optimizer.zero_grad()
```

## Backend and Ops

`SADL` builds on NumPy and CuPy. Which backend is used for operations, and which memory buffer backs a `Tensor`, depends on the device the `Tensor` is on.

| Device | Code | Backend |
|--------|-------------|-------------|
| CPU | `TensorDevice("cpu")` | Numpy |
| CUDA (GPU) | `TensorDevice("cuda", device_id=<id>)` | Cupy |

## Tensor

The [`Tensor`](../sadl/tensor.py) class is the central data primitive in `SADL`. All operations in `SADL` are performed on `Tensor`.

To perform operations on `Tensor`, use Python operators like `+`, `*`, and `@`, or NumPy functions like `np.sum`, `np.add`, and `np.matmul`.

Find more detailed information on what a `Tensor` in `SADL` actually is, including why NumPy operations work out of the box, [here](TENSOR.md).

Like in PyTorch, a `Tensor` can be created via the `Tensor` class directly, but this requires an existing `NDArray`:

```python
import cupy as cp
import numpy as np
import sadl
from sadl import NDArray

# cpu array, type annotation NDArray is not required, but added here just to be explicit
array: NDArray = np.array([1,2,3])

# or: array on gpu device 0
# with xp.cuda.Device(0):
#     array: NDArray = cp.array([1,2,3])

tensor = sadl.Tensor(array)
```

For simplicity, it is encouraged to use the factory function instead.
This allows you to pass in any data that can be converted into a NumPy array,
and also gives you control over which device the `Tensor` should be created on, as well as which datatype it should have.

```python
import cupy as cp
import numpy as np
import sadl
from sadl import NDArray

gpu = sadl.TensorDevice("cuda", device_id=0)  # device_id defaults to 0
cpu = sadl.TensorDevice("cpu")

tensor = sadl.tensor([1,2,3])  # default device is cpu
tensor2 = sadl.tensor(1, device=cpu)

nested_list = [[1,2,3],[2,3,4],[3,4,5]]
tensor3 = sadl.tensor(
    data=nested_list,
    device=gpu,
    dtype=np.float32,  # just use numpy dtypes
)
```

## Parameter

`Parameter` is a `Tensor` subclass for learnable weights. These are the trainable weights of any neural network
created with `SADL`. They can be used to, for example, represent the weight matrix of a linear layer.

Parameters must be created with the `sadl.Parameter` class directly.
`__init__` expects either a `Tensor` or `NDArray`.

```python
import numpy as np
import sadl

tensor = sadl.tensor([1,2,3])
param = sadl.Parameter(tensor)

array = np.array([1,2,3])
param_from_array = sadl.Parameter(array)

tensor = sadl.tensor([1,2,3], device=sadl.TensorDevice("cuda", device_id=1))
param_on_second_gpu = sadl.Parameter(tensor)
```

Parameters can be frozen by setting `requres_grad` to `False`. However, to avoid accidental
freezing `requres_grad` is not exposed as an argument in `Parameter.__init__` and must be set after construction:

```python
param_on_second_gpu.requires_grad = False
```

Since each `Parameter` is also a `Tensor`, it has the same properties and methods available.

## Disable gradient tracking

| Export | Description |
|--------|-------------|
| `no_grad` | Context manager disabling gradient tracking |
| `no_grad_fn` | Decorator disabling gradients in a function |

```python
with sadl.no_grad():
    y = model(x)  # No graph built

@sadl.no_grad_fn
def inference(model, x):
    return model(x)
```

## Functions

In `SADL`, components of neural networks are represented as `sadl.Function`, which can be seen as
an abstract mathematical function, matching how they appear in research papers.

A `Function` typically has multiple instances of `Parameter` as attributes,
which should be used in the `__call__` method. `__call__` defines the forward pass.

```python
import sadl
import numpy as np

# Linear layer on the CPU without a bias term
# -> Is on cpu, because "W" is created from a numpy array
class CPULinearWithoutBias(sadl.Function):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
    ) -> None:
        self.W = sadl.Parameter(np.random.rand(dim_in, dim_out))

    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:
        return np.matmul(x, self.W)  # or just "x @ self.W"

my_func = CPULinearWithoutBias(dim_in=3, dim_out=4)
x = sadl.tensor([[1,2,3]])  # is on cpu by default
y = my_func(x)
```

Every parameter a function has must either be a direct or indirect attribute of that class.
If a parameter should be optimized, it should be used in some way during the forward pass (`__call__`).

```python
import sadl
import numpy as np

class WeirdCPULinear(sadl.Function):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
    ) -> None:
        self.W = CPULinearWithoutBias(dim_in=dim_in, dim_out=dim_out)
        self.bias_dict = {
            "bias": sadl.Parameter(np.zeros((dim_out,))),
        }
        self.i_am_not_optimized = sadl.Parameter(np.zeros((10_000,)))


    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:
        return self.W(x) + self.bias_dict["bias"]

my_func = WeirdCPULinear(dim_in=3, dim_out=4)
x = sadl.tensor([[1,2,3]])  # is on cpu by default
y = my_func(x)
```

Parameters can be defined as direct or transitive attributes, or in containers like lists and dicts.

A function can also have no parameters. Standard examples are activation functions:
```python
import sadl
import numpy as np

class Softmax(sadl.Function):
    def __call__(self, x: sadl.Tensor) -> sadl.Tensor:
        x = np.exp(x)
        return x / np.sum(x, axis=-1, keepdims=True)

softmax = Softmax()
```

`sadl.Parameter` and `sadl.Function` are enough to create even complex transformer architectures.

## Optimizers

Optimizers perform, as the name suggests, the actual optimization of the `Function` parameters.
In `SADL`, they are represented by the class `sadl.Optimizer`.
Subclasses are the optimizer implementations, like the built-in `sadl.SGD` and `sadl.Adam`.

```python
import sadl

# sadl.Mlp is a subclass of sadl.Function
model = sadl.Mlp([
    sadl.Linear(dim_in=784, dim_out=784),
    sadl.Linear(dim_in=784, dim_out=10),
])

device = sadl.TensorDevice("cuda", device_id=0)
model = model.copy_to_device(device)

optimizer = sadl.SGD(params=model.get_parameters(), lr=1e-3)
optimizer = optimizer.copy_to_device(device=device)
```

## Serialization

`SADL` has a custom serialization format. Not because it is necessary,
as one could easily rely on existing (and much better) serializations from e.g. NumPy or PyTorch,
but because it helps to understand what is happening during serialization and how it works.

| Function | Description |
|----------|-------------|
| `save(data, file_path)` | Save tensor(s) to `.sadl` file. Supports passing Tensors on cuda. |
| `load(file_path)` | Load tensor(s) from `.sadl` file |

Supports single `Tensor` or `OrderedDict[str, Tensor]`.

**Note**: During serialization only the underlying memory buffers are stored. During deserialization fresh Tensors are created from the loaded buffers.
That means that extra metadata stored in the `Tensor` will not be saved.

Loading and saving data pairs well with `get_parameters` and `load_parameters`, which both handle `OrderedDict[str, Tensor]`.
This is almost exactly the same as `state_dict` in PyTorch.

```python
import sadl

model = sadl.Mlp([
    sadl.Linear(dim_in=784, dim_out=784),
    sadl.Linear(dim_in=784, dim_out=10),
])

# Save the model parameter data
sadl.save(model.get_parameters(), file_path="your_path/model.sadl")

same_model_arch = sadl.Mlp([
    sadl.Linear(dim_in=784, dim_out=784),
    sadl.Linear(dim_in=784, dim_out=10),
])

# Load the model parameter data
model_state = sadl.load(file_path="your_path/model.sadl")

# Load the parameters back into the model
same_model_arch = same_model_arch.load_parameters(model_state)
```

Read more on persistence [here](DISK.md).


## End-to-End Example

An end-to-end example can be found in [this](../notebooks/mnist_demo.ipynb) notebook.
