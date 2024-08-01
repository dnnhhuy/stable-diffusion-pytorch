import torch
from torch import nn
from typing import Tuple
import pathlib
import numpy as np
import os
from torchvision import transforms
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
        
        