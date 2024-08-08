from models.diffusion import StableDiffusion
from models.ema import EMA
import torch
from torch import nn
from tqdm.auto import tqdm
from utils import datasets
import os
from torchvision import transforms
import numpy as np
import argparse

def train_step(model: nn.Module,
               ema_model: EMA,
               train_dataloader: torch.utils.data.DataLoader,
               device: torch.device,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module,
               uncondition_prob: float,
               use_ema: bool=False):
    
    train_loss = 0.
    model.train()
    
    for i, (imgs, labels) in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        imgs = imgs.to(device)
        
        # CFG: Unconditional pass
        if np.random.random() < uncondition_prob:
           labels = torch.zeros(labels.shape)
        labels = labels.type(torch.float32).to(device)
            
        loss = model(imgs, labels, loss_fn=loss_fn)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if ema_model is not None:
            ema_model.step(model)

        if i % 100 == 0:
            print(f"\nStep [{i}/{len(train_dataloader)}] | Loss: {loss.item():.2f}")
    
    train_loss /= len(train_dataloader)
    return train_loss
        
def test_step(model: nn.Module,
              test_dataloader: torch.utils.data.DataLoader, 
              device: torch.device, 
              loss_fn: nn.Module,
              tokenizer=None):
    
    test_loss = 0.

    model.eval()
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_dataloader, position=0, leave=True), leave=True):
            imgs = imgs.to(device)
            labels = labels.argmax(dim=1) + 1
            labels = labels.to(device)
            
            loss = model(imgs, labels, loss_fn=loss_fn)

            test_loss += loss.item()
            
    test_loss /= len(test_dataloader)
    return test_loss


def train(model: nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          epochs: int, 
          device: torch.device, 
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          save_dir: str,
          checkpoint_dir: str,
         use_ema: bool=False):
    
    results = {'train_loss': [],
              'test_loss': []}

    ema_model = None
    if use_ema:
        ema_model = EMA(model=model, beta=0.995)
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                ema_model=ema_model,
                                train_dataloader=train_dataloader, 
                                device=device,
                                uncondition_prob=0.2,
                                optimizer=optimizer,
                               loss_fn=loss_fn)

        test_loss = test_step(model=model,
                              test_dataloader=test_dataloader,
                              device=device,
                             loss_fn=loss_fn)

        print(f"\nEpoch {epoch+1} | "
        f"Train loss: {train_loss} | "
        f"Test loss: {test_loss}")

        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        
        if epoch % 5 == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss}, os.path.join(checkpoint_dir, f"stable_duffusion_epoch_{epoch}.ckpt"))

    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(save_dir, "stable_diffusion_final.ckpt"))
    print("Model saved at: ", os.path.join(save_dir, "stable_diffusion_final.ckpt"))
    return results

NUM_WORKERS = os.cpu_count()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--device', default='cpu', type=str, help='Choose device to train')
    parser.add_argument('--data_dir', default='data/sprites', type=str, help='Data directory')
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size")
    parser.add_argument('--use_ema', default=False, type=bool, help='Toggle to use EMA for training')
    parser.add_argument('--save_dir', default='./checkpoint/', help='Directory to save model')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/', help='Directory to save checkpoint')
    
    
    args = parser.parse_args()
    
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((16, 16))])

    train_dataloader, test_dataloader, num_classes = datasets.create_dataloaders(data_dir=args.data_dir, transform=transform, train_test_split=0.8, batch_size=args.batch_size, num_workers=NUM_WORKERS)
    model = StableDiffusion(model_type='class2img', num_classes=num_classes).to(args.device)
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()
    
    train(model, train_dataloader, test_dataloader, epochs=300, device=args.device, optimizer=optimizer, loss_fn=loss_fn, save_dir=args.save_dir, checkpoint_dir=args.checkpoint_dir)
    