import numpy as np

from activations import ReLU, Softmax
from layers import Linear

class FullyConnectedNeuralNetwork():
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)
        
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.x4 = None
        
        
    def forward(self, x):
        self.x1 = self.fc1(x)
        self.x2 = self.relu(self.x1)
        self.x3 = self.fc2(self.x2)
        self.x4 = self.softmax(self.x3)
        
        return self.x4
    
    def backward(self, dx): 
        dx4 = self.softmax.backward(dx)
        dx3 = self.fc2.backward(dx4)
        dx2 = self.relu.backward(dx3)
        dx1 = self.fc1.backward(dx2)
    
    def update(self, lr):
        self.fc2.update(lr)
        self.fc1.update(lr)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)  