import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from models.ddpm import DDPMSampler
from models.ddim import DDIMSampler
from models.vae import VAE, VQVAE
from models.unet import UNet
from models.cond_encoder import TextEncoder, ClassEncoder
import numpy as np
from PIL import Image
from typing import Tuple
import gc
from transformers import PreTrainedTokenizerFast

def scale_img(x: torch.Tensor, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = torch.clamp(x, new_min, new_max)
    return x

class StableDiffusion(nn.Module):
    def __init__(self, model_type: str, num_classes: int=None, vae_type: str=''):
        super().__init__()
        
        if vae_type == 'vqvae':
            self.vae = VQVAE()
        else:
            self.vae = VAE()
        
        if model_type == 'txt2img':
            self.cond_encoder = TextEncoder()
            self.unet = UNet(in_channels=4, out_channels=4, cond_dim=768)
        elif model_type == 'class2img':
            self.cond_encoder = ClassEncoder(num_classes=num_classes)
            self.unet = UNet(in_channels=4, out_channels=4, cond_dim=768)
        else:
            raise ValueError('Only support txt2img or class2img model types')
    
    def _preprocess_image(self, img: Image, img_size: Tuple[int, int], dtype: torch.dtype=torch.float32):
        img = img.resize(img_size)
        img = np.array(img)
        img = torch.tensor(img, dtype=dtype)
        img = scale_img(img, (0, 255), (-1, 1))
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        return img
    
    def generate(self, input_image: Image.Image,
                 img_size: Tuple[int, int],
                 prompt: str,
                 uncond_prompt: str,
                 do_cfg: bool,
                 cfg_scale: int,
                 device: torch.device,
                 strength:float,
                 inference_steps: int,
                 sampler: str,
                 use_cosine_schedule: bool,
                 seed: int,
                 tokenizer: PreTrainedTokenizerFast, 
                 batch_size: int,
                 gr_progress_bar=None,
                 ) -> np.ndarray:
        """Generate an image.

        Args:
            input_image (Image.Image): Input image if use image to generate another image
            img_size (Tuple[int, int]): Size of image
            prompt (str): Prompt to generate image
            uncond_prompt (str): Unconditional prompt
            do_cfg (bool): Whether to use CFG
            cfg_scale (int): Set classifer-free guidance scale (larger value tends to focus on conditional prompt,
                            smaller value tends to focus on unconditional prompt)
            device (torch.device): Device for inference
            strength (float): Set the strength to generate the image (Given image from the user, the smaller value
                            tends to generate an image closer to the original one)
            inference_steps (int): Step to generate an image
            sampler (str): Sampling method: 2 options available: DDPM and DDIM
            use_cosine_schedule (bool): Activate using cosine function to generate beta values used for adding and remove noise
                                        from the image.
            seed (int): Specify seed for reproducibility
            tokenizer: tokenizer

        Returns:
            np.ndarray: numpy array of the image
        """

        dtype = torch.get_default_dtype()

        img_h, img_w = img_size
        LATENT_HEIGHT, LATENT_WIDTH = img_h // 8, img_w // 8
        
        latent_shape = (batch_size, 4, LATENT_HEIGHT, LATENT_WIDTH)

        generator = torch.Generator(device=device)
        if not seed:
            generator.seed()
        else:
            generator.manual_seed(seed)

        if sampler == 'ddpm':
            sampler = DDPMSampler(use_cosine_schedule=use_cosine_schedule, device=device)
            # Set desired inference steps
            sampler._set_inference_steps(inference_steps)
        elif sampler == 'ddim':
            sampler = DDIMSampler(use_cosine_schedule=use_cosine_schedule, device=device)
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
                cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77, truncation=True).input_ids, dtype=torch.long, device=device).repeat(batch_size, 1)
                uncond_tokens = torch.tensor(tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77, truncation=True).input_ids, dtype=torch.long, device=device).repeat(batch_size, 1)
                context = torch.cat([cond_tokens, uncond_tokens], dim=0)
                context_embedding = self.cond_encoder(context)
            else:
                cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77, truncation=True).input_ids, dtype=torch.long, device=device)
                context_embedding = self.cond_encoder(cond_tokens)
            
            self.cond_encoder.to('cpu')
            
            # Encoding Image
            self.vae.to(device)
            if input_image:
                transformed_img = self._preprocess_image(input_image, img_size, dtype=torch.get_default_dtype()).to(device)

                encoder_noise = torch.randn((1, *latent_shape[1:]), generator=generator, device=device)
                latent_features, _, _ =  self.vae.encode(transformed_img, encoder_noise)
                
                sampler.set_strength(strength=strength)
                latents_noise = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
                latent_features, _ = sampler.forward_process(latent_features, sampler.timesteps[0].unsqueeze(0), latents_noise)
            else:
                latent_features = torch.randn(latent_shape, generator=generator, dtype=dtype, device=device)
                
            self.vae.to('cpu')
            
            # Denoising
            if gr_progress_bar is not None:
                timesteps = gr_progress_bar.tqdm(sampler.timesteps.to(device))
            else:
                timesteps = tqdm(sampler.timesteps.to(device))
            self.unet.to(device)
            
            for i, timestep in enumerate(timesteps):
                timestep = timestep.unsqueeze(0)
                # (b, 8, latent_height, latent_width)
                model_input = latent_features
                if do_cfg:
                    model_input = model_input.repeat(2, 1, 1, 1)
                    
                model_input = model_input.to(device)
        
                pred_noise = self.unet(model_input, timestep, context_embedding)
                
                if do_cfg:
                    cond_output, uncond_output = pred_noise.chunk(2)
                    pred_noise = cfg_scale * (cond_output - uncond_output) + cond_output
                
                latent_features = sampler.reverse_process(latent_features, timestep, pred_noise)
            
            self.unet.to('cpu')
    
            self.vae.to(device)
            generated_imgs = self.vae.decode(latent_features.to(device))
            self.vae.to('cpu')
    
            generated_imgs = scale_img(generated_imgs, (-1, 1), (0, 255), clamp=True)
            generated_imgs = generated_imgs.permute(0, 2, 3, 1)
            generated_imgs = generated_imgs.to('cpu', torch.uint8).numpy()
            # Reset inference steps
            sampler._set_inference_steps(sampler.noise_step)
            
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        return list(generated_imgs)

   
    def inpaint(self, 
                input_image: Image.Image,
                mask: Image.Image,
                img_size: Tuple[int, int],
                prompt: str,
                uncond_prompt: str,
                do_cfg: bool,
                cfg_scale: int,
                device: torch.device,
                strength: float,
                inference_steps: int,
                sampler: str, 
                use_cosine_schedule: bool,
                seed: int,
                tokenizer: PreTrainedTokenizerFast,
                gr_progress_bar=None) -> np.ndarray:
        """Generate an image.

        Args:
            input_image (Image.Image): Input image if use image to generate another image
            mask (Image.Image): Image mask for inpainting
            img_size (Tuple[int, int]): Size of image
            prompt (str): Prompt to generate image
            uncond_prompt (str): Unconditional prompt
            do_cfg (bool): Whether to use CFG
            cfg_scale (int): Set classifer-free guidance scale (larger value tends to focus on conditional prompt,
                            smaller value tends to focus on unconditional prompt)
            device (torch.device): Device for inference
            strength (float): Set the strength to generate the image (Given image from the user, the smaller value
                            tends to generate an image closer to the original one)
            inference_steps (int): Step to generate an image
            sampler (str): Sampling method: 2 options available: DDPM and DDIM
            use_cosine_schedule (bool): Activate using cosine function to generate beta values used for adding and remove noise
                                        from the image.
            seed (int): Specify seed for reproducibility
            tokenizer: tokenizer

        Returns:
            np.ndarray: numpy array of the image
        """

        dtype = torch.get_default_dtype()

        img_h, img_w = img_size
        LATENT_HEIGHT, LATENT_WIDTH = img_h // 8, img_w // 8
        
        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        generator = torch.Generator(device=device)
        if not seed:
            generator.seed()
        else:
            generator.manual_seed(seed)

        if sampler == 'ddpm':
            sampler = DDPMSampler(use_cosine_schedule=use_cosine_schedule, device=device)
            # Set desired inference steps
            sampler._set_inference_steps(inference_steps)
        elif sampler == 'ddim':
            sampler = DDIMSampler(use_cosine_schedule=use_cosine_schedule, device=device)
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
                uncond_tokens = torch.tensor(tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
                
                context = torch.cat([cond_tokens, uncond_tokens], dim=0)
                context_embedding = self.cond_encoder(context)
    
            else:
                cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
                context_embedding = self.cond_encoder(cond_tokens)
            
            self.cond_encoder.to('cpu')
           
            transformed_imgs = self._preprocess_image(input_image, img_size, dtype=dtype).to(device)
            
            mask = mask.resize(img_size)
            mask = np.array(mask)
            mask = np.expand_dims(mask, axis=[0, 1])
            mask = torch.tensor(mask, device=device, dtype=dtype)
            
            # Downsample Mask
            downsampled_mask = F.interpolate(mask, scale_factor=1/8, mode='bicubic').to(device)
            downsampled_mask = scale_img(downsampled_mask, (0, 255), (0, 1))
            downsampled_mask = downsampled_mask.type(torch.bool)
            
            # Encoding Image
            self.vae.to(device)
            encoder_noise = torch.randn((1, *latent_shape[1:]), generator=generator, device=device)
            encoded_img, _, _ =  self.vae.encode(transformed_imgs, encoder_noise)
            
            latents_noise = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
            sampler.set_strength(strength)
            latent_features, _ = sampler.forward_process(encoded_img, sampler.timesteps[0].unsqueeze(0), latents_noise)

            # Noise for masked part
            noise_features = torch.randn(latent_shape, generator=generator, device=device,  dtype=dtype)
            latent_features = torch.where(downsampled_mask.repeat(1, latent_shape[1], 1, 1), noise_features, latent_features)
            
            self.vae.to('cpu')

            # Denoising
            if gr_progress_bar is not None:
                timesteps = gr_progress_bar.tqdm(sampler.timesteps.to(device))
            else:
                timesteps = tqdm(sampler.timesteps.to(device), leave=0, position=0)
            self.unet.to(device)
            for i, timestep in enumerate(timesteps):
                timestep = timestep.unsqueeze(0)
                
                # (b, latent_dim, latent_height, latent_width)
                model_input = latent_features.to(device)
                if do_cfg:
                    model_input = model_input.repeat(2, 1, 1, 1).to(device)
                    
                pred_noise = self.unet(model_input, timestep, context_embedding)
                
                if do_cfg:
                    cond_output, uncond_output = pred_noise.chunk(2)
                    pred_noise = cfg_scale * (cond_output - uncond_output) + cond_output

                # Add noise
                noised_orig_img, _ = sampler.forward_process(encoded_img, timestep, pred_noise)

                # Only denoise masked part
                latent_features = torch.where(~downsampled_mask.repeat(1, latent_shape[1], 1, 1), noised_orig_img, latent_features)
                latent_features = sampler.reverse_process(latent_features, timestep, pred_noise)
                  
            self.unet.to('cpu')
            self.vae.to(device)
            generated_imgs = self.vae.decode(latent_features.to(device))
            self.vae.to('cpu')

            generated_imgs = scale_img(generated_imgs, (-1, 1), (0, 255), clamp=True)
            generated_imgs = generated_imgs.permute(0, 2, 3, 1)
            generated_imgs = generated_imgs.to('cpu', torch.uint8).numpy()
    
            # Reset inference steps
            sampler._set_inference_steps(sampler.noise_step)
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        return generated_imgs[0]
            
    def forward(self, images: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module):

        device = images.device
        sampler = DDPMSampler()
        
        with torch.no_grad():
            text_embeddings = self.cond_encoder(labels)
            
            latent_features, mean, stdev = self.vae.encode(images)
            # Actual noise
            timesteps = sampler._sample_timestep(images.shape[0]).to(device)
            x_t, actual_noise = sampler.forward_process(latent_features, timesteps)

        # Predict noise
        pred_noise = self.unet(x_t, timesteps, text_embeddings)

        unet_loss = loss_fn(actual_noise, pred_noise)
        
        with torch.no_grad():
            pred_image = self.vae.decode(latent_features)
        
        # Total Loss
        loss = unet_loss
        
        return loss, pred_image