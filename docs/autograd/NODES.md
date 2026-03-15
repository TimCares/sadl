# Nodes

In `SADL`, tensors do not just hold data.
Once autodiff enters the picture, they also become the nodes of the computation graph.

That is the key step from "a thin wrapper around array data" to "something that can participate in training."

Every time an operation on tensors is executed, `SADL` creates a new result tensor.
If gradient tracking is active, that result tensor can also store enough information to later answer questions like:

- Which tensors was this value created from?
- Which backward function belongs to this operation?
- Which extra context from the forward pass is needed during backpropagation?

This is why in `SADL`, the computation graph is not a separate giant data structure living somewhere else.
The tensors themselves are the graph nodes.

This naturally leads to two next questions:

1. How exactly is that graph represented?
2. How is it traversed during backpropagation?

That is what the next docs are about.
