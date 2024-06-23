import math

class Minigrad:
    def __init__(self, data, _prevs=(), _op='', _label=''):
        self.data = data
        self.grad = 0.0

        self._backward = lambda: None
        self._prev = set(_prevs)
        self._op = _op
        self._label = _label
    
    def __add__(self, other):
        other = other if isinstance(other, Minigrad) else Minigrad(other)
        out = Minigrad(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Minigrad) else Minigrad(other)
        out = Minigrad(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Minigrad(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Minigrad(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        
        out._backward = _backward
        return out
    
    def relu(self):
        out = Minigrad(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Minigrad(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return self + -other
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return self / other

    def __repr__(self) -> str:
        return f"Minigrad(data={self.data}, label='{self._label}')"
