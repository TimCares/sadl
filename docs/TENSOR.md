# Tensor

The [`Tensor`](../sadl/tensor.py) class works as a wrapper around `NDArray`, which is just a type alias
for the numpy `np.ndarray` class.
The array is stored in the attribute `data` of the Tensor class, and all operations performed on `Tensor` are actually working on this `data`.
