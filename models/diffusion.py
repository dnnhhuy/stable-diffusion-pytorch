import torch
from torch import nn
import torchvision
from torchvision import transforms
from .vae import VAE
from .unet import UNet
from .cond_encoder import TextEncoder
import numpy as np
from tqdm.auto import tqdm
from .utils import denormalize_img
from PIL import Image


IMG_HEIGHT = 64
IMG_WIDTH = 64
Z_HEIGHT = 64 // 8
Z_WIDTH = 64 // 8

class StableDiffusion:
    def __init__(self, noise_step: int=1000, beta_start: float=1e-4, beta_end: float=0.02):
        self.betas = torch.linspace(beta_start, beta_end, noise_step, dtype=torch.float)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.noise_step = noise_step

        self.timesteps = torch.from_numpy(np.arange(0, noise_step)[::-1].copy())
        self.vae = VAE()
        self.unet = UNet()
        self.cond_encoder = TextEncoder()

    def _set_inference_step(self, inference_steps=50):
        self.inference_steps = inference_steps
        ratio = self.noise_step // self.inference_steps
        self.timesteps = torch.from_numpy((np.arange(0, self.inference_steps) * ratio).round()[::-1].copy().astype(np.int64))
        

    def _get_prev_timestep(self, timestep: int):
        prev_t = timestep - self.noise_step // self.inference_steps
        return prev_t

    def set_strength(self, strength: float=0.8):
        start_t = self.inference_steps - (self.inference_steps * strength)
        self.timesteps = self.timesteps[start_t:]

    # x_t ~ q(x_t | x_0) = N(x_t, sqrt(a_hat_t) * x_0, sqrt(1 - a_hat_t) * I)
    def forward_process(self, x_0: torch.Tensor, timestep: int):
        # x_0: (b, c, h, w)
        t = timestep
        # (1,) -> (1, 1, 1, 1)
        alpha_hat_t = self.alphas_hat[t, None, None, None].to(x_0.device)
              
        noise = torch.randn_like(x_0, dtype=torch.float32, device=x_0.device)
        latent = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise
        
        return latent, noise

    # x_(t-1) ~ p(x_(t-1) | x_t) = N(x_(t - 1), mu_theta(x_t, x_0), beta_tilda_t * I)
    # mu_theta(x_t, x-0)1/sqrt(alpha_hat_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_hat_t) * epsilon_t)
    # beta_tilda_t = (1 - alpha_hat_(t-1)) / (1 - alpha_t) * beta_t
    def reverse_process(self, x_t: torch.Tensor, timestep: int, model_output=torch.Tensor) -> torch.Tensor:
        t = timestep
        prev_t = self._get_prev_timestep(t)

        
        alpha_t = self.alphas[t, None, None, None].to(x_t.device)
        alpha_hat_t = self.alphas_hat[t, None, None, None].to(x_t.device)
        prev_alpha_hat_t = self.alphas_hat[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        prev_alpha_hat_t = prev_alpha_hat_t.to(x_t.device)
        
        mu = 1/torch.sqrt(alpha_t) * (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_hat_t) * model_output)

        stdev = 0
        if t > 0:
            variance = (1 - prev_alpha_hat_t) / (1 - alpha_hat_t) * self.betas[t]
            variance = torch.clamp(variance, min=1e-20)
            stdev = torch.sqrt(variance)
            
        noise = torch.randn_like(x_t, dtype=torch.float32, device=x_t.device)
        less_noise_sample = mu + stdev * noise
        return less_noise_sample
    
    def generate(self, input_image: Image, 
                 transforms: torchvision.transforms,
                 prompt: str,
                 uncond_promt: str,
                 do_cfg: bool,
                 cfg_scale: int,
                 device: torch.device,
                 strength:float,
                 inference_steps: int,
                 tokenizer=None) -> torch.Tensor:
        
        z_shape = (1, 8, Z_HEIGHT, Z_WIDTH)
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
    
            self._set_inference_step(inference_steps)
    
            # Encoding Image
            self.vae.to(device)
            if input_image:
                input_image = input_image.resize(IMG_HEIGHT, IMG_WIDTH)
                input_image = np.array(input_image)
                input_image = torch.from_array(input_image, dtype=torch.float32, device=device)
                input_image = input_image.unsqueeze(0)
                input_image = input_image.permute(0, 3, 1, 2)
                
                transformed_img = transforms(input_image)
                latent_features =  self.vae.encode(transformed_img)
    
                self.set_strength(strength=strength)
                latent_features = self.forward_process(latent_features, self.timesteps[0])
                
            else:
                latent_features = torch.randn(z_shape, dtype=torch.float32, device=device)
            self.vae.to('cpu')

            # Denoising
            timesteps = tqdm(self.timesteps.to(device))
            # x_t: torch.Tensor, 
            # timestep: int, 
            # model_output=torch.Tensor
    
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
    
                latent_features = self.reverse_process(latent_features, timestep, pred_noise)
            self.unet.to('cpu')
    
            self.vae.to(device)
            generated_imgs = self.vae.decode(latent_features)
            self.vae.to('cpu')
    
            generated_imgs = denormalize_img(generated_imgs, (-1, 1), (0, 255), clamp=True)
            generated_imgs = generated_imgs.permute(0, 2, 3, 1)
            generated_imgs = generated_imgs.to('cpu', torch.uint8).numpy()
            return generated_imgs[0]
    
        
