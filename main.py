import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
from data import MNISTDataset, Dataloader
from losses import CrossEntropyLoss
from train import Trainer
from model import FullyConnectedNeuralNetwork
    
if __name__ == '__main__':
    model = FullyConnectedNeuralNetwork(28*28, 128, 10)
    
    transform = lambda x: x/255.0
    dataset = MNISTDataset(r'MnistDataset\train.csv', transform=transform)
    train_dataset, test_dataset = dataset.split(train_size=0.8)
    
    train_loader = Dataloader(train_dataset, batch_size=32)
    criterion = CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        lr=0.01,
        epochs=10,
        visualize=True
    )
    
    trainer.fit(train_loader)
    
    test_dataset = Dataloader(test_dataset, batch_size=64)
    trainer.validate(test_dataset)
        
    
    