import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms



from .vae import VAE
from .unet import UNet
from .cond_encoder import TextEncoder
import numpy as np
from tqdm.auto import tqdm
from ..utils.utils import denormalize_img
from torch
from PIL import Image


IMG_HEIGHT = 512
IMG_WIDTH = 512
LATENT_HEIGHT = IMG_HEIGHT // 8
LATENT_WIDTH = IMG_WIDTH // 8

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

    def _preprocess_image(self, img: PIL.Image):
        transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((IMG_HEIGHT, IMG_WIDTH))])
        return transform(img)
        

    def _set_inference_step(self, inference_steps=50):
        self.inference_steps = inference_steps
        ratio = self.noise_step // self.inference_steps
        self.timesteps = torch.from_numpy((np.arange(0, self.inference_steps) * ratio).round()[::-1].copy().astype(np.int64))
        

    def _sample_timestep():
        idx = torch.randint(start=0, end=self.inference_steps)
        return self.timesteps[idx]
        
    def _get_prev_timestep(self, timestep: int):
        prev_t = timestep - self.noise_step // self.inference_steps
        return prev_t

    def set_strength(self, strength: float=0.8):
        start_t = self.inference_steps - int(self.inference_steps * strength)
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
                 prompt: str,
                 uncond_promt: str,
                 do_cfg: bool,
                 cfg_scale: int,
                 device: torch.device,
                 strength:float,
                 inference_steps: int,
                 tokenizer=None) -> torch.Tensor:
        
        latent_shape = (1, 8, LATENT_HEIGHT, LATENT_WIDTH)
        
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
    
            self._set_inference_step(inference_steps)
    
            # Encoding Image
            self.vae.to(device)
            if input_image:
                transformed_img = self._preprocess_image(input_image).to(device)
                
                latent_features, _, _ =  self.vae.encode(transformed_img)
    
                self.set_strength(strength=strength)
                latent_features, noise = self.forward_process(latent_features, self.timesteps[0])
                
            else:
                latent_features = torch.randn(latent_shape, dtype=torch.float32, device=device)
            self.vae.to('cpu')

            # Denoising
            timesteps = tqdm(self.timesteps.to(device))
            
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

    def train_step(self, 
              train_dataloader: torch.utils.data.DataLoader, 
              num_epochs: int,
              device: torch.device,
              cfg_scale: float,
              uncondition_prob: float,
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
            timestep = self._sample_timestep().int()
            
            x_t, actual_noise = self.forward_pass(latent_features, timestep)
            
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
    
                timestep = self._sample_timestep()
                
                x_t, actual_noise = self.forward_process(latent_features, timestep)
                pred_noise = self.unet(x_t, timestep, prompt_encoding)
    
                loss = nn.MSELoss(actual_noise, pred_noise)
    
                pred_output = self.vae.decode(pred_noise)
    
                loss += nn.MSELoss(pred_output, imgs) + 1/2 * torch.sum(1 + torch.log(stdev.pow(2)) - mean.pow(2) - stdev.pow(2))
    
                test_loss += loss
                
        test_loss /= len(test_dataloader)
        return test_loss
        