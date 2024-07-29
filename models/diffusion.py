import torch
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from .ddpm import DDPMSampler
from .ddim import DDIMSampler
from .vae import VAE
from .unet import UNet
from .cond_encoder import TextEncoder
import sys

sys.path.append("..")
from utils.utils import denormalize_img
from PIL import Image

IMG_HEIGHT = 512
IMG_WIDTH = 512
LATENT_HEIGHT = IMG_HEIGHT // 8
LATENT_WIDTH = IMG_WIDTH // 8

class StableDiffusion:
        def __init__(self):
            self.vae = VAE()
            self.unet = UNet()
            self.cond_encoder = TextEncoder()
        
        def _preprocess_image(self, img: Image):
            transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((IMG_HEIGHT, IMG_WIDTH))])
            return transform(img)
        
        def generate(self, input_image: Image,
                     prompt: str,
                     uncond_promt: str,
                     do_cfg: bool,
                     cfg_scale: int,
                     device: torch.device,
                     strength:float,
                     inference_steps: int,
                     sampler: str,
                     use_cosine_schedule: bool,
                     tokenizer=None) -> torch.Tensor:
            
            latent_shape = (1, 8, LATENT_HEIGHT, LATENT_WIDTH)
        
            if sampler == 'ddpm':
                sampler = DDPMSampler(use_cosine_schedule=use_cosine_schedule)
                # Set desired inference steps
                sampler._set_inference_steps(inference_steps)
            elif sampler == 'ddim':
                sampler = DDIMSampler(use_cosine_schedule=use_cosine_schedule)
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
                    transformed_img = self._preprocess_image(input_image).unsqueeze(0).to(device)
                    
                    latent_features, _, _ =  self.vae.encode(transformed_img)
        
                    sampler.set_strength(strength=strength)
                    latent_features, noise = sampler.forward_process(latent_features, sampler.timesteps[0])
                    
                else:
                    latent_features = torch.randn(latent_shape, dtype=torch.float32, device=device)
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
        
                generated_imgs = denormalize_img(generated_imgs, (-1, 1), (0, 255), clamp=True)
                generated_imgs = generated_imgs.permute(0, 2, 3, 1)
                generated_imgs = generated_imgs.to('cpu', torch.uint8).numpy()
        
                # Reset inference steps
                sampler._set_inference_steps(sampler.noise_step)
                
                return generated_imgs[0]
        
        def train_step(self, 
                  train_dataloader: torch.utils.data.DataLoader, 
                  device: torch.device,
                  optimizer: torch.optim.Optimizer,
                  tokenizer=None
                 ):
            
            self.cond_encoder.train()
            self.vae.train()
            self.unet.train()
        
            train_loss = 0.
            
            for i, (imgs, prompt) in tqdm(enumerate(train_dataloader)):
                imgs = self._preprocess_image(imgs).to(device)
        
                # Unconditional pass
                if np.random.random() < uncondition_prob:
                    prompt = ''
                    
                prompt_tokens = torch.tensor(tokenizer.batch_encode_plus(prompt, padding='max_length', max_length=77).ids, dtype=torch.long, device=device)
        
                prompt_encoding = self.cond_encoder(prompt_tokens)
        
                latent_features, mean, stdev = self.vae.encode(img)
        
                # Actual noise
                timestep = sampler._sample_timestep().int()
                
                x_t, actual_noise =  sampler.forward_process(latent_features, timestep)
                
                # Predict noise
                pred_noise = self.unet(x_t, timestep, prompt_encoding)
        
                loss = nn.MSELoss(actual_noise, pred_noise)
        
                pred_image = self.vae.decode(pred_noise)
        
                loss += nn.MSELoss(original_image, pred_image) + 1/2 * torch.sum(1 + torch.log(stdev.pow(2)) - mean.pow(2) - stdev.pow(2))
                train_loss += loss
                
                results['train_loss'].append(loss)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            train_loss /= len(train_dataloader)
            return train_loss
        
        def test_step(self, test_dataloader: torch.utils.data.DataLoader, device: torch.device, tokenizer=None):
            test_loss = 0.
            
            self.cond_encoder.eval()
            self.vae.eval()
            self.unet.eval()
            
            with torch.inference_mode():
                for i, (imgs, prompt) in tqdm(enumerate(test_dataloader)):
                    imgs = self._preprocess_image(imgs).to(device)
                    
                    prompt_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77)
                    
                    prompt_encoding = self.cond_encoder(prompt_tokens)
        
                    latent_features, mean, stdev = self.vae.encode(imgs)
        
                    timestep = sampler._sample_timestep()
                    
                    x_t, actual_noise = sampler.forward_process(latent_features, timestep)
                    pred_noise = self.unet(x_t, timestep, prompt_encoding)
        
                    loss = nn.MSELoss(actual_noise, pred_noise)
        
                    pred_output = self.vae.decode(pred_noise)
        
                    loss += nn.MSELoss(pred_output, imgs) + 1/2 * torch.sum(1 + torch.log(stdev.pow(2)) - mean.pow(2) - stdev.pow(2))
        
                    test_loss += loss
                    
            test_loss /= len(test_dataloader)
            return test_loss