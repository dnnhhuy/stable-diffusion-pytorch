import torch
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from .ddpm import DDPMSampler
from .ddim import DDIMSampler
from .vae import VAE
from .unet import UNet
from .cond_encoder import TextEncoder, ClassEncoder
import sys
import numpy as np
sys.path.append("..")
from utils.utils import scale_img
from PIL import Image
from typing import Tuple


class StableDiffusion(nn.Module):
    def __init__(self, model_type: str, num_classes: int=None):
        super().__init__()
        self.vae = VAE()
        self.unet = UNet()
        if model_type == 'txt2img':
            self.cond_encoder = TextEncoder()
        elif model_type == 'class2img':
            self.cond_encoder = ClassEncoder(num_classes=num_classes)
        else:
            raise ValueError('Only support txt2img or class2img model types')
        self.dequant = torch.ao.quantization.DeQuantStub()
    
    def _preprocess_image(self, img: Image, img_size: Tuple[int, int]):
        img = img.resize(img_size)
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32)
        img = scale_img(img, (0, 255), (-1, 1))
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        return img
    
    def generate(self, input_image: Image,
                 img_size: Tuple[int, int],
                 prompt: str,
                 uncond_promt: str,
                 do_cfg: bool,
                 cfg_scale: int,
                 device: torch.device,
                 strength:float,
                 inference_steps: int,
                 sampler: str,
                 use_cosine_schedule: bool,
                 seed: int,
                 tokenizer=None) -> torch.Tensor:

        img_h, img_w = img_size
        LATENT_HEIGHT, LATENT_WIDTH = img_h // 8, img_w // 8
        
        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        generator = torch.Generator(device=device)
        if not seed:
            generator.seed()
        else:
            generator.manual_seed(seed)

        if sampler == 'ddpm':
            sampler = DDPMSampler(generator, use_cosine_schedule=use_cosine_schedule)
            # Set desired inference steps
            sampler._set_inference_steps(inference_steps)
        elif sampler == 'ddim':
            sampler = DDIMSampler(generator, use_cosine_schedule=use_cosine_schedule)
            sampler._set_inference_steps(inference_steps)
        else:
            raise ValueError("Invalid sampler, available sampler is ddpm or ddim")
        
        self.vae.eval()
        self.unet.eval()
        self.cond_encoder.eval()
        
        with torch.inference_mode():
            # Encoding Condition
            self.cond_encoder.to(device)
            
                
            if do_cfg:
                cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
                uncond_tokens = torch.tensor(tokenizer.batch_encode_plus([uncond_promt], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
    
                context = torch.cat([cond_tokens, uncond_tokens], dim=0)
                context_embedding = self.cond_encoder(context)
    
            else:
                cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
                context_embedding = self.cond_encoder(cond_tokens)
            
            self.cond_encoder.to('cpu')
           
            
            # Encoding Image
            self.vae.to(device)
            if input_image:
                transformed_img = self._preprocess_image(input_image, img_size).to(device)

                encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
                latent_features, _, _ =  self.vae.encode(transformed_img, encoder_noise)
                
                sampler.set_strength(strength=strength)
                latent_features, noise = sampler.forward_process(latent_features, sampler.timesteps[0])
            else:
                latent_features = torch.randn(latent_shape, generator=generator, dtype=torch.float32, device=device)
            self.vae.to('cpu')
            
            # Denoising
            timesteps = tqdm(sampler.timesteps.to(device))
            self.unet.to(device)
            for i, timestep in enumerate(timesteps):
                # (b, 8, latent_height, latent_width)
                model_input = latent_features
                if do_cfg:
                    model_input = model_input.repeat(2, 1, 1, 1)
                pred_noise = self.unet(model_input, timestep, context_embedding)
                
                if do_cfg:
                    cond_output, uncond_output = pred_noise.chunk(2)
                    pred_noise = cfg_scale * (cond_output - uncond_output) + uncond_output
                
                latent_features = sampler.reverse_process(latent_features, timestep, pred_noise)
            
            self.unet.to('cpu')
    
            self.vae.to(device)
            generated_imgs = self.vae.decode(latent_features)
            self.vae.to('cpu')
    
            generated_imgs = scale_img(generated_imgs, (-1, 1), (0, 255), clamp=True)
            generated_imgs = generated_imgs.permute(0, 2, 3, 1)
            generated_imgs = generated_imgs.to('cpu', torch.uint8).numpy()
    
            # Reset inference steps
            sampler._set_inference_steps(sampler.noise_step)
            
            return generated_imgs[0]
    
    def forward(self, image: torch.Tensor, label: torch.Tensor, loss_fn: nn.Module):

        device = image.device

        generator = torch.Generator(device=image.device)
        sampler = DDPMSampler(generator)

        prompt_encoding = self.cond_encoder(label)

        latent_features, mean, stdev = self.vae.encode(image)
        # Actual noise
        with torch.no_grad():
            timestep = sampler._sample_timestep().int().to(device)
            
            x_t, actual_noise = sampler.forward_process(latent_features, timestep)
        
        # Predict noise
        pred_noise = self.unet(x_t, timestep, prompt_encoding)

        unet_loss = loss_fn(pred_noise, actual_noise)

        pred_image = self.vae.decode(pred_noise)

        vae_loss = loss_fn(pred_image, image) + 1/2 * torch.sum(1 + torch.log(stdev.pow(2)) - mean.pow(2) - stdev.pow(2))
        
        loss = unet_loss + vae_loss
        
        return loss