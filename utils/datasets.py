import torch
from torch import nn
from typing import Tuple
import pathlib
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, transform:transforms.Compose=None):
        super().__init__()
        self.imgs, self.labels = self.load_data(data_dir)
        self.num_classes = int(self.labels.shape[1])
        self.transform = transform
        
    def load_data(self, data_dir: str):
        img_paths = os.path.join(data_dir, 'sprites.npy')
        label_paths = os.path.join(data_dir, 'sprites_labels.npy')
        imgs = np.load(img_paths)
        labels = np.load(label_paths)
        return imgs, labels

    def get_image(self, index: int):
        return self.imgs[index]

    def get_label(self, index: int):
        return self.labels[index]
        
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        img = self.get_image(index)
        
        if self.transform:
            img = self.transform(img)
            
        label = self.get_label(index)
        return img, label

def create_dataloaders(data_dir, 
                       train_test_split: float, 
                       transform: transforms.Compose, 
                       batch_size: int, 
                       num_workers: int):

    dataset = CustomDataset(data_dir, transform=transform)

    
    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader, dataset.num_classes
    
    
        
        