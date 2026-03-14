# About

Since I started my journey in machine learning, I have trained quite a diverse set of neural networks, from simple MLPs to large transformers with contrastive loss, knowledge distillation, and much more.

I was always quite familiar with how neural nets are trained and how backpropagation and gradient descent work on the surface:

- Do a forward pass
- Compute the loss
- Do the backward pass and figure out the gradient of each parameter w.r.t the loss
- Do a single gradient descent step

But if someone had asked me to build all of this from scratch, I would have struggled:
- How do I know which parameter contributed to the loss function?
- What even is a parameter? How do I represent it?
- How do I compute the gradient and where do I store it?
- How do I store which computations were performed by the model?

If I had just inspected the code of PyTorch or tinygrad, I might eventually have figured out some core logic,
although even finding the actual implementation of the core concepts is already a challenge in itself due to the size of these codebases.
And micrograd by Andrej Karpathy? Brilliant, but maybe a bit too minimal if you want to understand how more complex frameworks like PyTorch work.

```text
"What I cannot create, I do not understand" - Richard Feynman
```

To me, this means that true understanding comes from rebuilding concepts from scratch. It emphasizes active creation over passive learning.

A typical disease of people is LGTM: "Looks good to me." You see an existing solution to a problem and understand it, or maybe you can even verify it, which gives you the illusion that you could have created it yourself.
But is that really true? No one has ever learned programming just by looking at code.

This reminded me of my time at university, where I graded exams of undergrad students in C/C++. One glance was often enough to tell whether a student had actually done the exercises, or had just looked at the solutions and thought: "Yep, makes sense, I would have done the same." The result: a failed exam.

Especially now, when we can use AI to create software from scratch and start verifying code more than writing it, this problem has become more present than ever before.

There is even more to it: I remember Andrej Karpathy saying in a podcast that by building things yourself, you gain exposure to topics and problems you might never have encountered otherwise.
This will inevitably lead to struggle, but in my, admittedly not yet very long, experience, that is exactly where the learning happens.
To sum it up: if you start (re-)building things yourself, you will probably learn more than you originally set out to learn.

So, back to this project: all of these questions I would have struggled with were swirling in my mind, paired with the fact that I had already trained so many models, yet still knew so little about how training them actually works at the core.
Together with the thoughts of Feynman and Karpathy, I knew I had to build it from scratch.

The result is `SADL`: a minimal deep learning framework that is actually readable, and gives a glimpse into how more complex deep learning frameworks work.

Dive into the concept of `SADL` [here](CONCEPT.md).
