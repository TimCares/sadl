# Get Started

A small guide about all parts of `SADL`.

## Quick Start

`SADL` works similar to Pytorch.

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
optimizer = sadl.SGD(list(model.parameters), lr=0.01)
optimizer = optimizer.copy_to_device(gpu)

# A single training step
output = model(x)
loss = output.sum()
optimizer.backward(loss)
optimizer.step()
optimizer.zero_grad()
```

## Backend and Ops

`SADL` builds on numpy and cupy. Which backend is used for operations and the backing memory buffer depends on the device the Tensor is on.

| Device | Code | Backend |
|--------|-------------|-------------|
| CPU | `TensorDevice("cpu")` | Numpy |
| CUDA (GPU) | `TensorDevice("cuda", device_id=<id>)` | Cupy |

## Tensor

The [`Tensor`](../sadl/tensor.py) class is the central data holding primitive. All operations in `SADL` are being performed on `Tensor`.

**`To perform operations on Tensors you must either use python operators like "+", "*", @", ..., or numpy member functions, like "np.sum", "np.add", "np.matmul".`**

Find detailed information on what a `Tensor` in `SADL` actually is, including why numpy operations work out of the box, [here](TENSOR.md).

Like in Pytorch, a `Tensor` can be created via the `Tensor` class directly, but this requires an existing `NDArray`:

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
This allows you to pass in any data that can be converted into a numpy array,
and also gives you control on which device the Tensor/array should be created, as well as which datatype it should have.

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

**Properties**:
- `device`: The current device the Tensor is on, is of type `sadl.TensorDevice`.
- `shape`: Shape of the Tensor.
- `ndim`: Number of dimensions the Tensor has.
- `size`: Number of elements in the Tensor.
- `dtype`: The datatype of the Tensor, represented as a numpy dtype.

**Methods:**
- `astype(np.dtype)`: Create a copy of the Tensor with a different numpy dtype.
- `copy_to_device(device)`: Copy Tensor to CPU or any available GPU device.
- `detach()`: Remove from computation graph.
- `cpu()`: Move to CPU.
- `gpu(device=TensorDevice("cuda"))`: Move to GPU with device id 0.
- `item()`: If the Tensor is a scalar value, get the actual scalar value, e.g. float.

## Parameter

Tensor subclass for learnable weights. These are the traininable weights of any Neural Network
created with `SADL`. They can be used to e.g. represent the weight matrix of a linear layer.

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

Since each `Parameter` is also a `Tensor` they have the same properties and methods available.


## Factory Functions

| Function | Description |
|----------|-------------|
| `tensor(data, *, dtype, device, requires_grad, keep_grad)` | Create tensor on specified device |
| `ones_like(other, *, dtype, requires_grad)` | Tensor of ones matching shape |
| `zeros_like(other, *, dtype, requires_grad)` | Tensor of zeros matching shape |

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

In `SADL`, components of Neural Networks are represented as `sadl.Function`, which can be seen as
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
        self.W = sadl.Parameter(np.random.rand(self.dim_in, self.dim_out))

    def __call__(x: sadl.Tensor) -> sadl.Tensor:
        return np.matmul(x, self.W)  # or just "x @ self.W"

my_func = CPULinearWithoutBias(dim_in=3, dim_out=4)
x = sadl.tensor([[1,2,3]])  # is on cpu by default
y = my_func(x)
```

**Every** parameter a function has **must** either be a direct or indirect attribute of that class.
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
            "bias": sadl.Parameter(np.zeros((self.dim_out,))),
        }
        self.i_am_not_optimized = sadl.Parameter(np.zeros((10_000,)))


    def __call__(x: sadl.Tensor) -> sadl.Tensor:
        return np.matmul(x, self.W) + self.bias_dict["bias"]

my_func = WeirdCPULinear(dim_in=3, dim_out=4)
x = sadl.tensor([[1,2,3]])  # is on cpu by default
y = my_func(x)
```

**`Parameters can be defined as direct attributes, transitive, or in containers like lists and dicts`.**

A function can also have no parameters. A standard example are activation functions:
```python
import sadl
import numpy as np

class Softmax(sadl.Function):
    def __call__(x: sadl.Tensor) -> sadl.Tensor:
        x = np.exp(x)
        return x / np.sum(x, axis=-1, keepdims=True)

softmax = Softmax()
softmax.device  # "()" -> empty tuple, has no parameters and is device independent
```

`sadl.Parameter` and `sadl.Function` are enough to create even the most complex Transformers architectures.

**Properties:**
- `parameters`: View of all Parameters.
- `requires_grad`: Get/set gradient tracking for all parameters.
- `device`: Tuple of devices where parameters reside.
- `is_training`: If the model is currently in training mode (important for Dropout, BatchNorm, ...).

**Methods:**
- `get_parameters()`: Get OrderedDict of parameters.
- `load_parameters(parameters, match_function_device, partial)`: Load parameter values.
- `copy_to_device(device)`: Move the function/model to a device. Internally, moves all parameters to the device.
- `train()`: Set to training mode (important for Dropout, BatchNorm, ...).
- `inference()`: Set to inference mode (important for Dropout, BatchNorm, ...).


## Optimizers

Optimizers perform, as the name suggest, the actual optimization of the `Function` parameters.
In `SADL`, they are represented by the class `sadl.Optimizer`.
Subclasses are the optimizers implementations, like the build-in `sadl.optimizer.SGD` and `sadl.optimizer.Adam`.

```python
import sadl

# sadl.Mlp is a subclass of sadl.Function
model = sadl.Mlp([
    sadl.Linear(dim_in=784, dim_out=784),
    sadl.Linear(dim_in=784, dim_out=10),
])

optimizer = sadl.optimizer.SGD(params=list(model.parameters), lr=1e-3)
optimizer = optimizer.copy_to_device(device=DEVICE)
```

**Properties:**
- `state`: View of optimizer state tensors
- `device`: Tuple of devices where state resides

**Methods:**
- `backward(loss)`: Compute gradients via backpropagation with respect to a loss.
- `step()`: Take one step by updating the tracked parameters.
- `zero_grad(additional_tensors=None)`: Clear gradients.
- `get_state(to_device=None)`: Get OrderedDict of the optimizer state (important if the Optimizer has an internal state, e.g. for Adam).
- `load_state(state, match_device, partial)`: Load the state.
- `copy_to_device(device)`: Move state to device. Does nothing if no state is there, e.g. for vanilla SGD.


## Serialization

`SADL` has a custom serialization format. Not because it is necessary,
as one could easily rely on existing (and much better) serializations from e.g. numpy or Pytorch,
but because it helps to understand what is happening during serialization and how it works.

| Function | Description |
|----------|-------------|
| `save(data, file_path)` | Save tensor(s) to `.sadl` file. Supports passing Tensors on cuda. |
| `load(file_path)` | Load tensor(s) from `.sadl` file |

Supports single `Tensor` or `OrderedDict[str, Tensor]`.

**Note**: During serialization only the underlying memory buffers are stored. During deserialization fresh Tensors are created from the loaded buffers.
That means that no extra metadata stored in the Tensor will be safed!

Read more on that [here](DISK.md).


## End-to-End Example

Can be found in [this](../notebooks/mnist_demo.ipynb) notebook.
