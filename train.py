import numpy as np
from visualizer import NeuralNetworkVisualizer
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self, model, criterion, lr, epochs, visualize=False):
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        
        # Initialize visualizer
        self.visualizer = NeuralNetworkVisualizer(
            input_dim=784,  # 28*28 for MNIST
            hidden_dim=128, 
            output_dim=10
        ) if visualize else None
        
    def train(self, dataloader):
        total_loss = 0
        correct = 0
        total = 0
        
        # Update visualization with current weights
        if self.visualizer is not None:
            self.visualizer.update_weights(
                self.model.fc1.weights,
                self.model.fc2.weights
            )
        
        for images, labels in dataloader:
            labels = self._one_hot(labels, 10)
            images = images.reshape(-1, 28*28)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            dy = self.criterion.backward(outputs, labels)
            self.model.backward(dy)
            self.model.update(self.lr)
        
            total_loss += loss    
            total += len(labels)
            correct += self._get_correct_predictions(outputs, labels)
        
        avg_loss = total_loss/total
        accuracy = correct/total
        return avg_loss, accuracy
        
    def fit(self, train_dataloader):
        if self.visualizer is not None:
            plt.ion()  # Turn on interactive mode
        for epoch in range(self.epochs):
            avg_loss, accuracy = self.train(train_dataloader)
            print(f'Epoch: {epoch+1}, Loss: {avg_loss}, Accuracy: {accuracy}')
            if self.visualizer is not None:
                plt.pause(0.1)  # Add small pause to allow visualization to update
        if self.visualizer is not None:
            plt.ioff()  # Turn off interactive mode
            self.visualizer.show()  # Keep the final plot visible
        
    def validate(self, dataloader):
        correct = 0
        total = 0
        for images, labels in dataloader:
            labels = self._one_hot(labels, 10)
            images = images.reshape(-1, 28*28)
            outputs = self.model(images)
            correct += self._get_correct_predictions(outputs, labels)
            
            total += len(labels)
            
        accuracy = correct/total
        
        print(f'Test Accuracy: {accuracy}')
            
    def _get_correct_predictions(self, outputs, one_hot_labels):
        return np.sum(np.argmax(outputs, axis=1) == np.argmax(one_hot_labels, axis=1))
    
    def _one_hot(self, labels, num_classes):
        one_hot_labels = np.zeros((labels.size, num_classes))
        one_hot_labels[np.arange(labels.size), labels] = 1
        return one_hot_labels