import torch
from torch import nn
from typing import Tuple
import pathlib

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir=str, transform=None):
        self.paths = list(pathlib.Path(data_dir).glob('*/.jpg'))
        self.transform = transform


    def load_img(self, index: int): 
        

    def __len__(self):

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        