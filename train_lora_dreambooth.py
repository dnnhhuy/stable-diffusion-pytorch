import math
import gc
import os
import argparse
from tqdm.auto import tqdm
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import CLIPTokenizer
from utils import load_model, datasets
from models import get_lora_model, enable_lora
from models.ddim import DDIMSampler
from models.ema import EMA
from models.diffusion import StableDiffusion

def train_step(model: StableDiffusion,
               ema_model: EMA,
               train_dataloader: torch.utils.data.DataLoader,
               device: torch.device,
               optimizer: torch.optim.Optimizer,
               epoch: int,
               tokenizer: CLIPTokenizer, 
               gradient_accumulation_steps: int,
               train_text_encoder: bool = False) -> float:
    """Training Step
        Feed images into model and calculate loss.

    Args:
        model (StableDiffusion): Object Instance of model
        ema_model (EMA): Specify Instance of Object EMA model if use EMA
        train_dataloader (torch.utils.data.DataLoader): Train dataloader
        device (torch.device): Target device for training
        optimizer (torch.optim.Optimizer): Optimizer
        epoch (int): current epoch
        tokenizer (CLIPTokenizer): Tokenizer
        gradient_accumulation_steps (int): Step for accumulating gradients

    Returns:
        float: Value of loss
    """
    
    prior_loss_weight = 1.0
    
    train_loss = 0.0
    
    # Turn off gradients for untrained parts of the model
    model.vae.requires_grad_(False)
    if not train_text_encoder:
        model.cond_encoder.requires_grad_(False)
    
    model.unet.to(device)
    model.unet.train()
    if train_text_encoder:
        model.cond_encoder.to(device)
        model.cond_encoder.train()
    
    
    pbar = tqdm(train_dataloader, leave=True, position=0, desc=f"Epoch {epoch}", ncols=100)
    for i, batch in enumerate(pbar):
        imgs = batch['pixel_values'].to(device)
        
        prompt_tokens = torch.tensor(tokenizer.batch_encode_plus(batch['prompts'], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
        
        sampler = DDIMSampler()
        
        if train_text_encoder:
            text_embeddings = model.cond_encoder(prompt_tokens)
            
            with torch.no_grad():
                model.vae.to(device)
                latent_features, mean, stdev = model.vae.encode(imgs)
                model.vae.to("cpu")
        else:
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
        loss = F.mse_loss(pred_instance_noise.float(), actual_instance_noise.float(), reduction="mean")
        
        # Class Prior Preservation Loss
        loss += F.mse_loss(pred_class_prior_noise.float(), actual_class_prior_noise.float(), reduction="mean") * prior_loss_weight
        
        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        if ((i + 1) % gradient_accumulation_steps == 0) or (i + 1 == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
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
              test_dataloader: DataLoader, 
              device: torch.device,
              tokenizer: CLIPTokenizer) -> float:
    """Evaluate step

    Args:
        model (nn.Module): Instance object of main model
        test_dataloader (DataLoader): Test dataloader
        device (torch.device): Target device for evaluating
        tokenizer (CLIPTokenizer): tokenizer

    Returns:
        float: Test loss
    """
    
    prior_loss_weight = 1.
    
    test_loss = 0.
    model.eval()
    
    pbar = tqdm(test_dataloader, leave=True, position=0, desc=f"Evaluating:", ncols=100)
    for i, batch in enumerate(pbar):
        imgs = batch['pixel_values'].to(device)
        
        prompt_tokens = torch.tensor(tokenizer.batch_encode_plus(batch['prompts'], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
        
        device = imgs.device
        sampler = DDIMSampler()
        
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
            
            test_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    test_loss /= len(train_dataloader)
    return test_loss


def train(model: StableDiffusion, 
          tokenizer: CLIPTokenizer,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          max_train_steps: int,
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
         use_flash_attn: bool=False,
         train_text_encoder: bool=False) -> Dict:
    
    results = {'train_loss': [],
              'test_loss': []}

    ema_model = None
    if use_ema:
        ema_model = EMA(model=model, beta=0.995)
        
    if gradient_checkpointing:
        model.unet.gradient_checkpointing_enabled(enabled=True)
        if train_text_encoder:
            model.cond_encoder.gradient_checkpointing_enabled(enabled=True)
        
    if use_flash_attn:
        model.unet.enable_flash_attn()
    
    
    writer = SummaryWriter(log_dir="./runs/")
    
    # Calculate epochs based on max_train_steps
    num_update_steps_per_epochs = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    epochs = math.ceil(max_train_steps / num_update_steps_per_epochs)
    
    # Start training
    for epoch in range(start_epoch, epochs):
        train_loss = train_step(model=model,
                                tokenizer=tokenizer,
                                ema_model=ema_model,
                                train_dataloader=train_dataloader, 
                                device=device,
                                optimizer=optimizer,
                                epoch=epoch,
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                train_text_encoder=train_text_encoder)
        lr_scheduler.step(train_loss)  
         
        test_loss = test_step(model=model,
                              test_dataloader=test_dataloader,
                              device=device,
                              tokenizer=tokenizer)
        
        print(f"Train Loss: {train_loss} | Test Loss: {test_loss}| Current LR: {lr_scheduler.get_last_lr()}")
        
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)
        
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        
        if use_lora:
            save_weights = {}
            if use_ema:
                for name, params in model.named_parameters():
                    if params.requires_grad:
                        save_weights[name] = ema_model.ema_model.get_parameter(name)
            else:
                for name, params in model.named_parameters():
                    if params.requires_grad:
                        save_weights[name] = params
                    
            torch.save({
            'epoch': epoch,
            'model_state_dict': save_weights,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss}, os.path.join(checkpoint_dir, f"stable_diffusion_lora_epoch_{epoch}.ckpt"))
        else:
            if use_ema:
                torch.save({
                'epoch': epoch,
                'model_state_dict': ema_model.ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss}, os.path.join(checkpoint_dir, f"stable_diffusion_epoch_{epoch}.ckpt"))
            else:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss}, os.path.join(checkpoint_dir, f"stable_diffusion_epoch_{epoch}.ckpt"))
            
    writer.close()
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
    parser.add_argument('--use_ema', metavar="", action=argparse.BooleanOptionalAction, help='Toggle to use EMA for training')
    parser.add_argument('--save_dir', default='./checkpoints/', help='Directory to save model')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/', help='Directory to save checkpoint')
    parser.add_argument('--pretrained_path', default=None, help='Pretrained model path')
    parser.add_argument('--lr', default=1e-6, type=float, help='Learning rate')
    parser.add_argument('--max_train_steps', default=1000, type=int, help='Max training steps')
    parser.add_argument('--use_lora', metavar="", action=argparse.BooleanOptionalAction, help='Option to use LoRA in training')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=bool, help="Graddient accumulation steps")
    parser.add_argument('--gradient_checkpointing', metavar="", action=argparse.BooleanOptionalAction, help="Apply gradient checkpointing")
    parser.add_argument('--use_flash_attn', metavar="", action=argparse.BooleanOptionalAction, help="Option to use Flash Attention")
    parser.add_argument('--train_text_encoder', metavar="", action=argparse.BooleanOptionalAction, help="Train text encoder")
    parser.add_argument('--use_8bit_adam', metavar="", action=argparse.BooleanOptionalAction, help="Use 8-bit Adam")
    
    args = parser.parse_args()
    model, tokenizer = load_model(args)
    
    if args.use_lora:
        model.unet = get_lora_model(model.unet,
                                    rank=32, 
                                    alphas=32, 
                                    lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out'])
        model.unet = enable_lora(model.unet, 
                                 lora_modules=['proj_q', 'proj_k', 'proj_v', 'proj_out'],
                                 enabled=True)
    
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # optimizer_class = torch.optim.SGD
        
    optimizer = optimizer_class(params=model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Define lr scheduler
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1, last_epoch=-1)
    
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
          max_train_steps=args.max_train_steps,
          device=args.device, 
          optimizer=optimizer, 
          lr_scheduler=lr_scheduler, 
          save_dir=args.save_dir, 
          checkpoint_dir=args.checkpoint_dir, 
          start_epoch=start_epoch,
          use_lora=args.use_lora,
          gradient_accumulation_steps=args.gradient_accumulation_steps,
          gradient_checkpointing=args.gradient_checkpointing,
          use_flash_attn=args.use_flash_attn,
          train_text_encoder=args.train_text_encoder)