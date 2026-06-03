# micrograd

A tiny scalar-valued autograd engine and a small neural network library built on
top of it. I wrote this from scratch to understand how backpropagation and
neural network training actually work, following Andrej Karpathy's micrograd.

Everything operates on single scalars rather than tensors. It's slow, but the
point is clarity: you can read every line and see exactly how gradients flow.

## What's here

- `micrograd/engine.py` — the `Value` class. Wraps a scalar, tracks the
  operations that produced it, and computes gradients with reverse-mode
  autodiff.
- `micrograd/nn.py` — `Neuron`, `Layer`, and `MLP` classes for building small
  multi-layer perceptrons on top of `Value`.
- `micrograd.ipynb` — the notebook where I worked through it: numerical
  derivatives, manual backpropagation through a single neuron, drawing the
  expression graph, and finally training an MLP with gradient descent.

## The engine

A `Value` records its data, its gradient, and the children that produced it.
Supported operations: addition, subtraction, multiplication, division, power,
`exp`, and `tanh`. Each operation knows how to send the gradient back to its
inputs.

```python
from micrograd.engine import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + 1.0
c.backward()

print(a.grad)  # da/dc = -3.0
print(b.grad)  # db/dc = 2.0
```

Calling `backward()` builds a topological ordering of the graph, then walks it
in reverse, applying the chain rule at each node.

## The neural net

`Neuron`, `Layer`, and `MLP` are thin wrappers around `Value`. An `MLP` is just
a list of layers, and each layer is a list of neurons.

```python
from micrograd.nn import MLP

n = MLP(3, [4, 4, 1])   # 3 inputs, two hidden layers of 4, one output
x = [2.0, 3.0, -1.0]
n(x)
```

### Training

Training is a plain gradient descent loop: forward pass, compute the loss, zero
the gradients, backpropagate, then nudge every parameter against its gradient.

```python
for k in range(20):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
```

## Running it

Clone the repo and import directly from the root:

```python
from micrograd.engine import Value
from micrograd.nn import MLP
```

The notebook also uses `numpy`, `matplotlib`, and `graphviz` for plotting and
visualizing the expression graph.
