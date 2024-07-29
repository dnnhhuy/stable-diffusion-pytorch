import torch
from torch import nn
import numpy as np


class DDPMSampler:
    def __init__(self, noise_step: int=1000, beta_start: float=1e-4, beta_end: float=0.02, use_cosine_schedule: bool=True):
        self.betas = torch.linspace(beta_start, beta_end, noise_step, dtype=torch.float)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.noise_step = noise_step
        
        # Cosine-based noise schedule
        if use_cosine_schedule:
            s = 1e-9
            f_t = lambda t: np.cos((t/noise_step + s)/(1 + s) * np.pi/2) ** 2
            self.alphas_hat = f_t(torch.arange(0, noise_step + 1)) / f_t(0)
            self.betas = torch.clip(1 - self.alphas_hat[1:]/self.alphas_hat[:-1], 0, 0.999)
            self.alphas = 1 - self.betas
            self.alphas_hat = self.alphas_hat[1:]
            
        

        self.timesteps = torch.from_numpy(np.arange(0, noise_step)[::-1].copy())        

    def _set_inference_steps(self, inference_steps=50):
        self.inference_steps = inference_steps
        step = self.noise_step // self.inference_steps
        self.timesteps = torch.from_numpy((np.arange(0, self.inference_steps) * step).round()[::-1].copy().astype(np.int64))
        

    def _sample_timestep():
        idx = torch.randint(start=0, end=self.timesteps[0])
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
        