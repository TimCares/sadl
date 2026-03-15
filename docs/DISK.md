# Disk (Serialization)

Only the Tensors underlying memory buffer, which are exactly numpy arrays, are safed.
That means the gradient, and any information about the Tensor in the computation graph (see [here](autograd/COMPUTATION_GRAPH.md) for more on that), and generally and extra metadata are not retained.

This is done on purpose, as all other information must be configured when the actual Tensor instance is there.
In the case of Functions: Whether the model is e.g. in training mode, or on which device it is, is decided only are runtime and in which environment the model is loaded anyways.
