from models.diffusion import StableDiffusion
from models.ema import EMA
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import datasets
import os
from torchvision import transforms
import numpy as np

def train_step(model: nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               device: torch.device,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module,
               uncondition_prob: float,
               use_ema: bool=False):
    
    train_loss = 0.
    
    model.train()
    
    if use_ema:
        ema = EMA(beta=0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad(False)
        
    with torch.autograd.set_detect_anomaly(True):
        for i, (imgs, labels) in tqdm(enumerate(train_dataloader)):
            imgs = imgs.to(device)
            
            labels = labels.argmax(dim=1) + 1
            labels = labels.to(device)
        
            # CFG: Unconditional pass
            if np.random.random() < uncondition_prob:
               labels = torch.zeros(labels.shape)
                
            loss = model(imgs, labels.int(), loss_fn=loss_fn)
            train_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_ema:
                ema_model.step(ema_model, model)
    
    train_loss /= len(train_dataloader)
    return train_loss
        
def test_step(model: nn.Module,
              test_dataloader: torch.utils.data.DataLoader, 
              device: torch.device, 
              loss_fn: nn.Module,
              tokenizer=None):
    
    test_loss = 0.

    model.eval()
    
    with torch.inference_mode():
        for i, (imgs, labels) in tqdm(enumerate(test_dataloader)):
            imgs = imgs.to(device)
            labels = labels.argmax(dim=1) + 1
            
            loss = model(imgs, labels, loss_fn=loss_fn)

            test_loss += loss
            
    test_loss /= len(test_dataloader)
    return test_loss


def train(model, train_dataloader, test_dataloader, epochs, device, optimizer, loss_fn):
    
    results = {'train_loss': [],
              'test_loss': []}
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                train_dataloader=train_dataloader, 
                                device=device,
                                uncondition_prob=0.2,
                                optimizer=optimizer,
                               loss_fn=loss_fn)

        test_loss = test_step(model=model,
                              test_dataloader=test_dataloader,
                              device=device,
                             loss_fn=loss_fn)

        print(f"Epoch {epoch+1} | "
        f"Train loss: {train_loss} | "
        f"Test loss: {test_loss}")

        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

    return results

NUM_WORKERS = os.cpu_count()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((64, 64))])
    dataset = datasets.CustomDataset('data/sprites', transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)
    model = StableDiffusion(model_type='class2img', num_classes=dataset.num_classes).to(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    train(model, train_dataloader, test_dataloader, epochs=300, device=device, optimizer=optimizer, loss_fn=loss_fn)
    