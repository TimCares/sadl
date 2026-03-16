# Disk (Serialization)

`SADL` has its own binary serialization format. Not because it is necessary, as one could easily rely on NumPy's `.npy`, Python's pickle, or PyTorch's `torch.save`. But because building it from scratch helps to understand what serialization of tensor data actually means and how little is required.

## What Gets Saved

Only the underlying memory buffer of each Tensor is saved. That is it.

No gradient. No computation graph metadata. No `requires_grad` flag. No `is_training` state. No device information.

This is done on purpose.

All of that metadata is configuration that belongs to the runtime environment, not to the data on disk. Whether a model is in training mode, on which device it runs, or which parameters are frozen are decisions made when the model is loaded and used, not when it is stored.

The only things that are persisted are:

- the raw array bytes
- the dtype
- the shape
- an optional string key (for `OrderedDict` entries)

That is the minimum needed to reconstruct a Tensor.

## The `OrderedDict` Approach

`save` and `load` support two forms of data:

1. A single `Tensor`
2. An `OrderedDict[str, Tensor]`

The second form is the important one. It gives structure to the data on disk: each tensor has a name, and the order is preserved.

This pairs naturally with the rest of `SADL`:

- `model.get_parameters()` returns an `OrderedDict[str, Parameter]`
- `optimizer.get_state()` returns an `OrderedDict[str, Tensor]`

So saving and loading a model is just:

```python
import sadl

# Save
sadl.save(model.get_parameters(), "model.sadl")

# Load
state = sadl.load("model.sadl")
model.load_parameters(state)
```

The same string keys that `get_parameters` produces are the keys that end up on disk. No mapping layer, no index remapping. The names you see in memory are the names you see in the file.

## Device Handling

When saving, all Tensors are automatically moved to CPU first. This means you can save a model that lives on GPU without worrying about it.
This is more efficient the because the data is moved to CPU dynamically, and not once in a batch.

When loading, you can specify a target device:

```python
state = sadl.load("model.sadl", device=sadl.TensorDevice("cuda", 0))
```

If no device is specified, data is loaded to CPU.

## The `.sadl` Format

Files must use the `.sadl` extension. Both `save` and `load` enforce this.

The binary layout is straightforward:

| Field | Type | Description |
|-------|------|-------------|
| Magic | 4 bytes | `b"SADL"` |
| Version | uint8 | Format version (currently 1) |
| Num tensors | uint32 | How many tensors follow |

Then for each tensor:

| Field | Type | Description |
|-------|------|-------------|
| Key length | uint32 | Length of the UTF-8 key string |
| Key | bytes | The key string |
| Dtype length | uint8 | Length of the dtype name |
| Dtype | bytes | NumPy dtype name (e.g. `"float32"`) |
| Ndim | uint8 | Number of dimensions |
| Shape | ndim x uint64 | One uint64 per dimension |
| Data | raw bytes | C-contiguous array data |

Everything is simple. There is no compression, no alignment padding, no nested structure beyond the flat sequence of tensors.

When a single `Tensor` is saved (not an `OrderedDict`), it is stored under the key `__single__`. On load, if that is the only key present, the single Tensor is returned directly instead of a dict.

The format version is separate from the library's semantic version. It only gets bumped when the serialization code changes in a backwards-incompatible way.

## API

| Function | Description |
|----------|-------------|
| `save(data, file_path)` | Save a `Tensor` or `OrderedDict[str, Tensor]` to a `.sadl` file. |
| `load(file_path, device=None)` | Load from a `.sadl` file. Returns a `Tensor` or `OrderedDict[str, Tensor]`. |

## Summary

The serialization in `SADL` is deliberately minimal. It saves raw array data with just enough metadata to reconstruct the tensors. Everything else, gradients, graph context, device placement, training mode, is left to the code that loads the data.

This keeps the format simple, portable, and easy to understand.

Further reading:
- What a Tensor actually stores -> [`TENSOR.md`](TENSOR.md)
- How `get_parameters` and `load_parameters` work -> [`FUNCTION.md`](FUNCTION.md)
- How the optimizer state is saved -> [`OPTIMIZER.md`](OPTIMIZER.md)
