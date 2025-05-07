import os
import gc

from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from typing import Tuple, Union
from transformers import PreTrainedTokenizerFast

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.scheduler import DDPMSampler, DDIMSampler
from models.vae import VAE
from models.clip import OpenCLIP
from models.unet import UNet

def denormalize(x: torch.Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)

class StableDiffusion(nn.Module):
    def __init__(self, sampler: str='ddpm', use_cosine_schedule: bool=False, device: str='cpu'):
        super().__init__()
        self.vae = VAE()
        self.clip = OpenCLIP()
        self.unet = UNet()
        
        if sampler == 'ddpm':
            self.sampler = DDPMSampler(use_cosine_schedule=use_cosine_schedule, device=device)
        elif sampler == 'ddim':
            self.sampler = DDIMSampler(use_cosine_schedule=use_cosine_schedule, device=device)
        else:
            raise ValueError("Invalid sampler, available sampler is ddpm or ddim")
        
    @staticmethod
    def from_pretrained(pretrained_dir, device: str = 'cpu', sd_version: str = "1.5"):
        model = StableDiffusion(device=device)
        model.vae = VAE.from_pretrained(os.path.join(pretrained_dir, "vae"), device=device)
        model.clip = OpenCLIP().from_pretrained(os.path.join(pretrained_dir, "text_encoder"), device=device)
        model.unet = UNet.from_pretrained(os.path.join(pretrained_dir, "unet"), device=device,  sd_version=sd_version)
        return model
    
    def _preprocess_image(self, img: Image, img_size: Tuple[int, int], dtype: torch.dtype=torch.float32):
        image_transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        img = image_transforms(img)
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        return img
    
    @torch.no_grad()
    def generate_in_one_step(self, input_image: Image.Image,
                 img_size: Tuple[int, int],
                 prompt: str,
                 uncond_prompt: str,
                 do_cfg: bool,
                 cfg_scale: int,
                 device: torch.device,
                 sampler: str,
                 use_cosine_schedule: bool,
                 seed: int,
                 tokenizer: PreTrainedTokenizerFast, 
                 batch_size: int,
                 ) -> np.ndarray:
        """
        Generate an image in one step.
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
        
        self.vae.eval()
        self.unet.eval()
        self.clip.eval()
        
        with torch.no_grad():
            # Encoding Condition
            self.clip.to(device)
            cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77, truncation=True).input_ids, dtype=torch.long, device=device)
            context_embedding = self.clip.text_model(cond_tokens)
            self.clip.to('cpu')
            
            # Encoding Image
            self.vae.to(device)
            latent_features = torch.randn(latent_shape, generator=generator, dtype=dtype, device=device)
            self.vae.to('cpu')
            
            # Denoising
            self.unet.to(device)
            max_timestep = sampler.timesteps[0].to(device)
            max_timestep = max_timestep.unsqueeze(0)
            
            # (b, 8, latent_height, latent_width)
            
            alpha_T, sigma_T = 0.0047 ** 0.5, (1 - 0.0047) ** 0.5
            pred_noise = self.unet(latent_features.to(device), max_timestep, context_embedding)
            pred_x0 = (latent_features - sigma_T * pred_noise) / alpha_T
            
            self.unet.to('cpu')
    
            self.vae.to(device)
            generated_imgs = self.vae.decode(pred_x0)
            self.vae.to('cpu')
            
            generated_imgs = (generated_imgs + 1) / 2
            
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        return list(generated_imgs)
    
    def generate(self, input_image: Image.Image,
                 img_size: Tuple[int, int],
                 prompt: str,
                 uncond_prompt: str,
                 do_cfg: bool,
                 cfg_scale: int,
                 device: torch.device,
                 strength:float,
                 inference_steps: int,
                 sampler: Union[DDIMSampler|DDPMSampler],
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
        sampler._set_inference_steps(inference_steps)
        img_h, img_w = img_size
        LATENT_HEIGHT, LATENT_WIDTH = img_h // 8, img_w // 8
        
        latent_shape = (batch_size, 4, LATENT_HEIGHT, LATENT_WIDTH)

        generator = torch.Generator(device=device)
        if not seed:
            generator.seed()
        else:
            generator.manual_seed(seed)

        self.vae.eval()
        self.unet.eval()
        self.clip.eval()
        
        with torch.no_grad():
            # Encoding Condition
            self.clip.to(device)
            if do_cfg:
                cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77, truncation=True).input_ids, dtype=torch.long, device=device).repeat(batch_size, 1)
                uncond_tokens = torch.tensor(tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77, truncation=True).input_ids, dtype=torch.long, device=device).repeat(batch_size, 1)
                prompt_emb = self.clip.text_model(cond_tokens)
                negative_prompt_emb = self.clip.text_model(uncond_tokens)
                context_embedding = torch.cat([negative_prompt_emb, prompt_emb], dim=0)
            else:
                cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77, truncation=True).input_ids, dtype=torch.long, device=device)
                context_embedding = self.clip.text_model(cond_tokens)
            self.clip.to('cpu')
            
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
                    uncond_output, cond_output = pred_noise.chunk(2)
                    pred_noise = uncond_output + cfg_scale * (cond_output - uncond_output)
                latent_features = sampler.reverse_process(latent_features, timestep, pred_noise)
            
            self.unet.to('cpu')
    
            self.vae.to(device)
            generated_imgs = self.vae.decode(latent_features.to(device))
            self.vae.to('cpu')

            # generated_imgs = denormalize(generated_imgs)
            # generated_imgs = generated_imgs.cpu().permute(0, 2, 3, 1).float().numpy()
            # generated_imgs = (generated_imgs * 255).round().astype("uint8")
            generated_imgs = (generated_imgs + 1) / 2
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
        self.clip.text_model.eval()
        
        with torch.inference_mode():
            # Encoding Condition
            self.clip.text_model.to(device)
            
                
            if do_cfg:
                cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
                uncond_tokens = torch.tensor(tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
                
                context = torch.cat([cond_tokens, uncond_tokens], dim=0)
                context_embedding = self.clip.text_model(context)
    
            else:
                cond_tokens = torch.tensor(tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids, dtype=torch.long, device=device)
                context_embedding = self.clip.text_model(cond_tokens)
            
            self.clip.text_model.to('cpu')
           
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
            text_embeddings = self.clip.text_model(labels)
            
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