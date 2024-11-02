import numpy as np
import pandas as pd



class MNISTDataset():
    def __init__(self, csv_file=None, transform=None):
        self.data = pd.read_csv(csv_file).to_numpy() if csv_file else None
        self.images = self.data[:, 1:].reshape(-1, 28, 28) if csv_file else None
        self.y_labels = self.data[:, 0] if csv_file else None
        
        self.transform = transform
        
    def alt_init(self, data, transform=None):   
        self.data = data
        self.images = self.data[:, 1:].reshape(-1, 28, 28)
        self.y_labels = self.data[:, 0]
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:  
            image = self.transform(image)
        
        return image, self.y_labels[idx]
    
    def split(self, train_size=0.8, shuffle=True):
        train_size = int(len(self.data) * train_size)
        
        indices = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_data = self.data[train_indices]
        test_data = self.data[test_indices]
        
        train_dataset = MNISTDataset()
        train_dataset.alt_init(train_data, self.transform)
        
        test_dataset = MNISTDataset()
        test_dataset.alt_init(test_data, self.transform)
        
        return train_dataset, test_dataset
        
        
        
class Dataloader():
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.dataset_indices = np.arange(len(dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if self.shuffle:
            np.random.shuffle(self.dataset_indices)
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            indices = self.dataset_indices[i:i+self.batch_size]
            yield self.dataset[indices]