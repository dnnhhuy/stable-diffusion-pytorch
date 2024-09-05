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
from torchinfo import summary
from models.lora import get_lora_model
from transformers import CLIPTokenizer
from utils.utils import load_model

def train_step(model: StableDiffusion,
               ema_model: EMA,
               train_dataloader: torch.utils.data.DataLoader,
               device: torch.device,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module,
               uncondition_prob: float,
               epoch: int,
               tokenizer: CLIPTokenizer):
    
    train_loss = 0.
    model.train()
    pbar = tqdm(train_dataloader, leave=True, position=0, desc=f"Epoch {epoch}", ncols=100)
    for i, (imgs, prompt) in enumerate(pbar):
        imgs = imgs.to(device)
        
        # CFG: Unconditional pass
        if np.random.random() < uncondition_prob:
           prompt = ['' for _ in range(imgs.shape[0])]
        prompt_tokens = torch.tensor(tokenizer.batch_encode_plus(prompt, padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
        
            
        loss, _ = model(imgs, prompt_tokens, loss_fn=loss_fn)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if ema_model is not None:
            ema_model.step(model)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    train_loss /= len(train_dataloader)
    return train_loss
        
def test_step(model: nn.Module,
              test_dataloader: torch.utils.data.DataLoader, 
              device: torch.device, 
              loss_fn: nn.Module,
              tokenizer: CLIPTokenizer):
    
    test_loss = 0.

    model.eval()
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(test_dataloader, position=0, leave=True)):
            imgs = imgs.to(device)
            
            labels = torch.tensor(tokenizer.batch_encode_plus([labels], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
            
            loss, _ = model(imgs, labels, loss_fn=loss_fn)

            test_loss += loss.item()
            
    test_loss /= len(test_dataloader)
    return test_loss


def train(model: StableDiffusion, 
          tokenizer: CLIPTokenizer,
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          epochs: int, 
          device: torch.device, 
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
          loss_fn: nn.Module,
          save_dir: str,
          checkpoint_dir: str,
          start_epoch: int=0,
         use_ema: bool=False):
    
    results = {'train_loss': [],
              'test_loss': []}

    ema_model = None
    if use_ema:
        ema_model = EMA(model=model, beta=0.995)
        
    for epoch in range(start_epoch, epochs):
        train_loss = train_step(model=model,
                                ema_model=ema_model,
                                train_dataloader=train_dataloader, 
                                device=device,
                                uncondition_prob=0.1,
                                optimizer=optimizer,
                               loss_fn=loss_fn,
                               epoch=epoch,
                               tokenizer=tokenizer)

        test_loss = test_step(model=model,
                              test_dataloader=test_dataloader,
                              device=device,
                             loss_fn=loss_fn,
                             tokenizer=tokenizer)

        lr_scheduler.step(test_loss)
        
        print(f"Train Loss: {train_loss} | Test Loss: {test_loss} | Current LR: {lr_scheduler.get_last_lr()}")

        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss}, os.path.join(checkpoint_dir, f"stable_diffusion_epoch_{epoch}.ckpt"))

    print("Saving model...")
    torch.save(model.state_dict(), os.path.join(save_dir, "stable_diffusion_final.ckpt"))
    print("Model saved at: ", os.path.join(save_dir, "stable_diffusion_final.ckpt"))
    return results

    
NUM_WORKERS = os.cpu_count()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--device', default='cpu', type=str, help='Choose device to train')
    parser.add_argument('--model_path', default='./weights/model/v1-5-pruned-emaonly.ckpt', help='Model path')
    parser.add_argument('--tokenizer_dir', default='./weights/tokenizer/', help='Tokenizer dir')
    parser.add_argument('--data_dir', default='data/sprites', type=str, help='Data directory')
    parser.add_argument('--img_size', default=32, type=int, help='Image size')
    parser.add_argument('--batch_size', default=32, type=int, help="Batch size")
    parser.add_argument('--use_ema', default=False, type=bool, help='Toggle to use EMA for training')
    parser.add_argument('--save_dir', default='./checkpoint/', help='Directory to save model')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/', help='Directory to save checkpoint')
    parser.add_argument('--pretrained_path', default=None, help='Pretrained model path')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--use_lora', default=False, type=bool, help='Option to use LoRA in training')

    args = parser.parse_args()

    train_dataloader, test_dataloader = datasets.create_dataloaders(data_dir=args.data_dir, img_size=(args.img_size, args.img_size), train_test_split=1.0, batch_size=args.batch_size, num_workers=NUM_WORKERS)

    model, tokenizer = load_model(args)
    
    if args.use_lora:
        model = get_lora_model(model)
        
    model = model.to(device=args.device)
        
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
    
    # Define Loss Function
    loss_fn = nn.MSELoss()

    start_epoch = 0
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    
    train(model, tokenizer, train_dataloader, test_dataloader, epochs=300, device=args.device, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_fn=loss_fn, save_dir=args.save_dir, checkpoint_dir=args.checkpoint_dir, start_epoch=start_epoch)
