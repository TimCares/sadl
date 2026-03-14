# Concept

Before reading this, one small note: the goal of this framework is by no means to show some novel new idea, and I do not like labeling things as "novel" anyway.
Instead, the goal is to show what amazing things one can build on existing components with minimal and clear code.
And if you look at how the code works, you will find that the core algorithms powering the training of today's foundation models really are not that difficult to understand.

A deep learning framework consists of 3 major components:

1. Linear Algebra (think add, subtract, matrix multiplication, ...)
2. Gradient computation
3. Backpropagation

Most deep learning frameworks, like PyTorch, TensorFlow, and tinygrad, implement all three components from scratch.
This makes sense if you are serious about building a framework used in enterprise-grade systems and for computationally extreme tasks, like training models with trillions of parameters in giant datacenters.

For these cases, you need extreme efficiency, support a lot of use cases, and a lot of low-level utility and infrastructure code, with C-level kernels,
multi-node gradient communication, and distributed checkpointing just to name a few.

There is even quite the amusing story where engineers at Meta had to compute dummy operations to avoid stressing the power grid supplying the
data center when training LLaMa 3 (find the story [here](https://newsletter.semianalysis.com/p/ai-training-load-fluctuations-at-gigawatt-scale-risk-of-power-grid-blackout)).

While supporting these complex concepts is certainly impressive and of great value, it is exactly this that inflates the size of these codebases by several orders of magnitude,
making it incredibly difficult to understand how the core of these systems actually works under the hood.

And this creates a tension for `SADL`: support a high variety of concepts so that it is actually usable in practice, or keep it focused on the core.

There are many things and concepts I could add over time, like new layers, more complete architectures, and domain-specific tooling.
But if all of that keeps landing in this repository, `SADL` would slowly run into exactly the same problem PyTorch and other frameworks have: the core concepts become diluted.

This is where two ideas come together:

1. Why not keep the core repository minimal and readable, and move higher-level functionality into separate repositories that behave like plugins or companion libraries?
That would be closer in spirit to how projects like `timm`, `torchvision`, or `torchaudio` extend PyTorch without bloating the already huge PyTorch core.

This keeps the core in this repo, which makes it easy to inspect and learn.

2. Why do we even have to implement Linear Algebra, C-level code, memory buffers, and all this extra logic from scratch?

While components 2. and 3. of deep learning frameworks are more specific and used mostly in machine learning, linear algebra is something much more general: it appears everywhere and is one of the most well-known branches of mathematics.
So why not use an existing linear algebra and array library, and focus on the training part of neural nets?

We set our sights on [`numpy`](https://numpy.org/doc/stable/index.html), one of the most popular scientific computing and array libraries in Python.
It already provides everything we need for component 1: memory buffers, arrays, operations on arrays, including element-wise ops, matrix multiplication, and linear algebra.
And the API? Extremely well known.
If you have used Python, chances are pretty high that you have already used NumPy or have at least heard of it.
This is an advantage, because most computer scientists are already very familiar with it.

## Using NumPy as the Backend

Building on NumPy means that `SADL` does not need to reinvent the array world.
Instead of designing a completely separate tensor API, it can stay very close to an interface most people already know.

That matters for two reasons:

1. It keeps the mental model simple.
If you know how arrays behave in NumPy, much of that intuition transfers directly to `SADL`.

2. It keeps the core code focused.
The library can spend its complexity budget on gradients, backpropagation, optimizers, and device handling instead of rebuilding general-purpose array infrastructure.

This is also why `Tensor` in `SADL` is deliberately a thin layer around array data rather than a giant abstraction.
The point is not to hide NumPy, but to build on top of it.

Care to learn more about the reusability of NumPy? Then head right on [here](BACKEND.md).
