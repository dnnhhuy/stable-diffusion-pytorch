import torch.utils
import torch.utils.data
from models.ddpm import DDPMSampler
from models.ema import EMA
from models.diffusion import StableDiffusion
import torch
from torch import nn
from tqdm.auto import tqdm
from utils import datasets
import os
import argparse
from torchinfo import summary
from models.lora import get_lora_model
from transformers import CLIPTokenizer
from utils.utils import load_model
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc


def train_step(model: StableDiffusion,
               ema_model: EMA,
               train_dataloader: torch.utils.data.DataLoader,
               device: torch.device,
               optimizer: torch.optim.Optimizer,
               epoch: int,
               tokenizer: CLIPTokenizer, 
               gradient_accumulation_steps: int, 
               gradient_checkpointing: bool,
               use_flash_attn: bool):
    
    prior_loss_weight = 1.
    
    train_loss = 0.
    model.eval()
    model.unet.train()
    if gradient_checkpointing:
        model.unet.gradient_checkpointing_enabled()
    
    if use_flash_attn:
        model.unet.enable_flash_attn()
        
    pbar = tqdm(train_dataloader, leave=True, position=0, desc=f"Epoch {epoch}", ncols=100)
    for i, batch in enumerate(pbar):
        imgs = batch['pixel_values'].to(device)
        
        prompt_tokens = torch.tensor(tokenizer.batch_encode_plus(batch['prompts'], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
        
        device = imgs.device
        sampler = DDPMSampler()
        
        with torch.no_grad():
            model.cond_encoder.to(device)
            text_embeddings = model.cond_encoder(prompt_tokens)
            model.cond_encoder.to("cpu")
            
            model.vae.to(device)
            latent_features, mean, stdev = model.vae.encode(imgs)
            model.vae.to("cpu")
            
            # Actual noise
            timesteps = sampler._sample_timestep(imgs.shape[0]).to(device)
            x_t, actual_noise = sampler.forward_process(latent_features, timesteps)
            
        actual_instance_noise, actual_class_prior_noise = actual_noise.chunk(2, dim=0)

        # Predict noise
        pred_noise = model.unet(x_t, timesteps, text_embeddings)
        pred_instance_noise, pred_class_prior_noise = pred_noise.chunk(2, dim=0)
        
        # Instance loss
        loss = F.mse_loss(pred_instance_noise, actual_instance_noise, reduction="mean")
        
        # Class Prior Preservation Loss
        loss += F.mse_loss(pred_class_prior_noise, actual_class_prior_noise, reduction="mean") * prior_loss_weight
        
        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if ((i + 1) % gradient_accumulation_steps == 0) or (i + 1 == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        if ema_model is not None:
            ema_model.step(model)

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
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          epochs: int,
          device: torch.device, 
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
          save_dir: str,
          checkpoint_dir: str,
          start_epoch: int=0,
         use_ema: bool=False,
         use_lora: bool=False,
         gradient_accumulation_steps: int=1,
         gradient_checkpointing: bool=False,
         use_flash_attn: bool=False):
    
    results = {'train_loss': [],
              'test_loss': []}

    ema_model = None
    if use_ema:
        ema_model = EMA(model=model, beta=0.995)
        
    for epoch in range(start_epoch, epochs):
        train_loss = train_step(model=model,
                                tokenizer=tokenizer,
                                ema_model=ema_model,
                                train_dataloader=train_dataloader, 
                                device=device,
                                optimizer=optimizer,
                                epoch=epoch,
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                gradient_checkpointing=gradient_checkpointing,
                                use_flash_attn=use_flash_attn)

        # lr_scheduler.step(train_loss)
        
        print(f"Train Loss: {train_loss} | Current LR: {lr_scheduler.get_last_lr()}")

        results['train_loss'].append(train_loss)
        
        if use_lora:
            save_weights = {}
            for name, params in model.named_parameters():
                if params.requires_grad:
                    save_weights[name] = params
                    
            torch.save({
            'epoch': epoch,
            'model_state_dict': save_weights,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss}, os.path.join(checkpoint_dir, f"stable_diffusion_lora_epoch_{epoch}.ckpt"))
        else:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss}, os.path.join(checkpoint_dir, f"stable_diffusion_epoch_{epoch}.ckpt"))

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
    parser.add_argument('--save_dir', default='./checkpoints/', help='Directory to save model')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/', help='Directory to save checkpoint')
    parser.add_argument('--pretrained_path', default=None, help='Pretrained model path')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--use_lora', default=False, type=bool, help='Option to use LoRA in training')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=bool, help="Graddient accumulation steps")
    parser.add_argument('--gradient_checkpointing', default=False, type=bool, help="Apply gradient checkpointing")
    parser.add_argument('--use_flash_attn', default=False, type=bool, help="Option to use Flash Attention")
    
    args = parser.parse_args()
    model, tokenizer = load_model(args)
    
    if args.use_lora:
        model = get_lora_model(model, rank=8, alphas=16)
        
    model = model.to(device=args.device)
        
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    
    train_dataloader, test_dataloader = datasets.create_dataloaders(instance_data_dir=os.path.join(args.data_dir, "instance_data"), 
                                                                    class_data_dir=os.path.join(args.data_dir, "class_prior_data"),
                                                                    train_test_split=1.0,
                                                                    batch_size=args.batch_size,
                                                                    num_workers=0,
                                                                    img_size=(args.img_size, args.img_size))
    

    
    train(model, 
          tokenizer, 
          train_dataloader, 
          test_dataloader, 
          epochs=300,
          device=args.device, 
          optimizer=optimizer, 
          lr_scheduler=lr_scheduler, 
          save_dir=args.save_dir, 
          checkpoint_dir=args.checkpoint_dir, 
          start_epoch=start_epoch,
          use_lora=args.use_lora,
          gradient_accumulation_steps=args.gradient_accumulation_steps,
          gradient_checkpointing=args.gradient_checkpointing,
          use_flash_attn=args.use_flash_attn)