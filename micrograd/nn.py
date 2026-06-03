import random

from micrograd.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, num_of_in):
        self.w = [Value(random.uniform(-1,1)) for _ in range(num_of_in)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        activation = sum((wi * xi for wi, xi in zip(self. w, x)), self.b)   # wi*xi + b
        out = activation.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    

class Layer(Module):
    def __init__(self, num_of_in, num_of_out):
        self.neurons = [Neuron(num_of_in) for _ in range(num_of_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    

class MLP(Module):  # Multi-Layer Perceptron
    def __init__(self, num_of_in, num_of_outs_list):    # i.e. MLP(3, [4, 4, 1])
        size = [num_of_in] + num_of_outs_list
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(num_of_outs_list))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]