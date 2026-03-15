# Autograd

**Note:** Before reading this, you should already have some rough intuition for backpropagation.
I warmly encourage you to watch [this](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) brilliant video series by Andrej Karpathy on backpropagation.

<hr>

In order to train a neural network, we need to find out how a parameter `w` of the model contributes to the loss, i.e. what its gradient is with respect to the loss function:

$$
\frac{\partial \mathcal{L}}{\partial w}
$$

In practice, that contribution is usually not direct.
The parameter `w` affects one value, that value affects the next one, and so on until we eventually arrive at the loss.

That is exactly why we need a computation graph.
A computation graph is simply a record of how each value was produced from earlier values.

In a neural network, every operation that eventually contributes to the loss has to be tracked:
adds, multiplies, matrix multiplications, activations, reductions, and everything in between.
If we do not remember that path, we cannot walk it backwards during backpropagation and apply the chain rule.

We can express that idea like this:

$$
\frac{\partial \mathcal{L}}{\partial w}
=
\frac{\partial \mathcal{L}}{\partial z_n}
*
\ldots
*
\frac{\partial z_2}{\partial z_1}
*
\frac{\partial z_1}{\partial w}
$$

Where `z` is some intermediate result.

<hr>

In `SADL`, all parameters `w` are represented by the `Parameter` class, a subtype of `Tensor`.
Intermediate results, and generally all other tracked values `z`, are represented by `Tensor`.

So the rest of the story is really just this:

1. How does a `Tensor` become a node?
2. How is the computation graph built while the forward pass runs?
3. How do we walk that graph backwards?
4. Where do the actual gradient formulas come from?

That is exactly what the next docs explain.

Further reading:
- What nodes are in `SADL` -> [`NODES.md`](NODES.md)
- How the graph is built during the forward pass -> [`COMPUTATION_GRAPH.md`](COMPUTATION_GRAPH.md)
- How backpropagation walks the graph -> [`BACKPROPAGATION.md`](BACKPROPAGATION.md)
- Where operation-specific gradients come from -> [`GRADIENTS.md`](GRADIENTS.md)
