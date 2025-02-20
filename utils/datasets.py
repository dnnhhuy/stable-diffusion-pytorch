import torch
from torch import nn
from typing import Tuple
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np
import os, sys
import random

sys.path.append("..")

def scale_img(x: torch.Tensor, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = torch.clamp(x, new_min, new_max)
    return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, img_size: Tuple[int, int]):
        super().__init__()
        self.imgs, self.labels = self.load_data(data_dir)
        self.num_classes = len(self.labels)
        self.img_size = img_size
        
    def load_data(self, data_dir: str):
        img_paths = os.path.join(data_dir, 'sprites.npy')
        label_paths = os.path.join(data_dir, 'sprites_labels.npy')
        imgs = np.load(img_paths)
        labels = np.load(label_paths)
        return imgs, labels

    def get_image(self, index: int):
        img = Image.fromarray(self.imgs[index])
        img = img.resize(self.img_size)
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32)
        img = scale_img(img, (0, 255), (-1, 1))
        img = img.permute(2, 0, 1)
        return img

    def get_label(self, index: int):
        return self.labels[index]
        
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        img = self.get_image(index)        
        label = self.get_label(index)
        return img, label
    
    
class DreamBoothDataset(torch.utils.data.Dataset):
    def __init__(self, instance_data_dir: str, class_data_dir: str, img_size: Tuple[int, int], num_class_prior_images: int = None):
        super().__init__()
        self.instance_imgs_path, self.instance_prompt = self.load_data(instance_data_dir)
        random.shuffle(self.instance_imgs_path)
        
        self.class_imgs_path, self.class_prompt = self.load_data(class_data_dir)
        self.class_imgs_path = self.class_imgs_path[:num_class_prior_images]
        
        self.img_size = img_size
        self.num_instance_imgs = len(self.instance_imgs_path)
        self.num_class_imgs = len(self.class_imgs_path)
        self.length = max(self.num_instance_imgs, self.num_class_imgs)
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        
    def load_data(self, data_dir: str):
        imgs_path = Path(data_dir).glob("*.jpg")
        with open((Path(data_dir) / "label.txt"), "r") as f:
            label = f.read()
        imgs = []
        for path in imgs_path:
            imgs.append(path)
        return imgs, label

    def transform_image(self, img: Image.Image):
        img = img.convert("RGB")
        return self.image_transforms(img)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        example = {}
        instance_img = Image.open(self.instance_imgs_path[index % self.num_instance_imgs]).convert("RGB")
        example['instance_img'] = self.transform_image(instance_img)
        example['instance_prompt'] = self.instance_prompt
        
        class_img = Image.open(self.class_imgs_path[index % self.num_class_imgs]).convert("RGB")
        example['class_img'] = self.transform_image(class_img)
        example['class_prompt'] = self.class_prompt
        
        return example
        

def collate_fn(examples):
    pixel_values = [example["instance_img"] for example in examples]
    pixel_values += [example["class_img"] for example in examples]
    
    prompts = [example["instance_prompt"] for example in examples]
    prompts += [example["class_prompt"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    
    batch = {"pixel_values": pixel_values,
             "prompts": prompts}
    
    return batch

def create_dataloaders(instance_data_dir, 
                       class_data_dir,
                       train_test_split: float,
                       batch_size: int, 
                       num_workers: int,
                      img_size: Tuple[int, int],
                      num_class_prior_images: int = None):

    dreambooth_dataset = DreamBoothDataset(instance_data_dir=instance_data_dir, class_data_dir=class_data_dir, img_size=img_size)

    
    train_size = int(train_test_split * len(dreambooth_dataset))
    test_size = len(dreambooth_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dreambooth_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda examples: collate_fn(examples))
    test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda examples: collate_fn(examples))

    return train_dataloader, test_dataloader
    
    
        
        