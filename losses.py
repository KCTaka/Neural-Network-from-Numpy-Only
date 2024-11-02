import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.output = None
        self.target = None
    
    def forward(self, x, y):
        self.output = x
        self.target = y
        return -np.mean(np.sum(y * np.log(x + 1e-7), axis=1))
    
    def backward(self, x, y):
        return (x - y)/x.shape[0]
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.forward(x, y)