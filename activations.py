import numpy as np

class ReLU():
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, dx):
        return dx * (self.input > 0)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
class Softmax():
    def __init__(self, dim = 0):
        self.dim = dim
        self.output = None
    
    def forward(self, x):
        self.output = np.exp(x) / np.sum(np.exp(x), axis=self.dim, keepdims=True)
        return self.output
    
    def backward(self, dx):
        return dx
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
