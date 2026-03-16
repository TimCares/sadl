# Device

## Why `TensorDevice`?

`SADL` supports two kinds of devices: CPU and CUDA (GPU).

CuPy, the library used for GPU arrays, has its own `cupy.cuda.Device` class.
But that class is tightly coupled to CUDA. It does not exist when only NumPy is installed, and it does not cover the CPU case at all.

`SADL` needs a single, uniform way to describe where a Tensor lives, regardless of whether that is a CPU or one of several GPUs.
That is why `TensorDevice` exists.

```python
import sadl

cpu = sadl.TensorDevice("cpu")
gpu0 = sadl.TensorDevice("cuda", device_id=0)
gpu1 = sadl.TensorDevice("cuda", device_id=1)
```

The name is intentional. It is not just `Device`, it is `TensorDevice`, because it describes the device on which a `Tensor` is located.
And it is not `.to(device)` like in PyTorch, it is `.copy_to_device(device)`, because that is what actually happens: a copy.

## `DeviceLike`

Not every place in the code requires a full `TensorDevice` object.
To make the API more convenient, `SADL` defines a `DeviceLike` type alias:

```python
DeviceLike = TensorDevice | Literal["cpu"] | int | SupportsCupyDevice
```

This means you can pass:

- a `TensorDevice` directly
- the string `"cpu"`
- an integer (interpreted as a CUDA device id)
- any object with an `id` attribute (like `cupy.cuda.Device`)

Internally, `TensorDevice.create(device)` normalizes any of these into a proper `TensorDevice`.

## Copying Tensors

In `SADL`, `copy_to_device` is a real operation, not just a utility.

For intermediate tensors in a computation graph, copying to another device is a **tracked operation**.
This means gradients can flow back through the copy during backpropagation.

That might sound surprising. Why would copying need to be differentiable?

Consider a sharded model where part of the computation happens on one GPU and part on another.
An activation produced on GPU 0 is copied to GPU 1 to continue the forward pass.
During the backward pass, the gradient needs to flow back from GPU 1 to GPU 0.
If the copy were not tracked, the gradient chain would break at the device boundary.

So for regular Tensors, `copy_to_device` participates in the computation graph like any other operation.

### Parameters Are Different

For `Parameter`, `copy_to_device` is **not** tracked.

Parameters are always leaves in the computation graph. Copying a parameter to a different device means moving the model, which is a setup step, not part of a forward pass.
There is no gradient to route back through a parameter copy, because there is no graph edge to follow: Parameters are always leaves!

That is why `Parameter.copy_to_device` creates a fresh `Parameter` with the same data on the new device, without any graph context.

## Why `device` Returns a Tuple

Both `Function.device` and `Optimizer.device` return `tuple[TensorDevice, ...]`, not a single `TensorDevice`.

The reason is sharding. A model's parameters might live on different devices.
If a `Linear` layer has its weight on GPU 0 and its bias on GPU 1, the function is on two devices.

```python
model.device  # (TensorDevice("cuda", 0), TensorDevice("cuda", 1))
```

In the common case where everything is on one device, the tuple has a single element.
But the type signature does not lie about the possibility.

## `TensorDevice` Is a Frozen Dataclass

`TensorDevice` is defined as a frozen dataclass.
This means it is immutable and hashable, so it can be used as a dictionary key or in a set.
Two `TensorDevice` instances with the same `type` and `device_id` are equal.

```python
sadl.TensorDevice("cuda", 0) == sadl.TensorDevice("cuda", 0)  # True
```

This is useful when collecting unique devices, which is exactly what `Function.device` does internally: it puts all parameter devices into a set.

## Summary

`TensorDevice` exists to give `SADL` a single, backend-independent way to talk about where data lives.
It is explicit, it is hashable, and it makes the naming transparent: `copy_to_device` says exactly what it does.

Further reading:
- How Tensors wrap the underlying arrays -> [`TENSOR.md`](TENSOR.md)
- How the backend selects NumPy or CuPy -> [`BACKEND.md`](BACKEND.md)
