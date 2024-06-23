import random
from minigrad.engine import Minigrad


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in, nonlin='relu'):
        self.w = [Minigrad(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Minigrad(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == 'relu':
            out = activation.relu()
        else:
            out = activation.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'Relu' if self.nonlin == 'relu' else 'Tanh'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, n_in, n_outs):
        size = [n_in] + n_outs
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
