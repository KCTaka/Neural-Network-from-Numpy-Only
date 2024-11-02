import numpy as np

class Linear():
    def __init__(self, input_dim, output_dim, bias=True):
        stdv = 1. / np.sqrt(input_dim)
        self.weights = np.random.uniform(-stdv, stdv, (output_dim, input_dim))
        self.bias = np.random.uniform(-stdv, stdv, (output_dim, 1)) if bias else None
        
        self.dweights = None
        self.dbias = None

        self.input = None
        
    def forward(self, x):
        self.input = x
        z = self.weights@x.T + self.bias
        return z.T
        
    def backward(self, dx):
        self.dweights = dx.T @ self.input
        self.dbias = np.sum(dx, axis=0, keepdims=True).T if self.bias is not None else None
        return dx @ self.weights
    
    def update(self, lr):
        self.weights -= lr * self.dweights
        self.bias = self.bias - lr * self.dbias if self.bias is not None else None
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)