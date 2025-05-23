import os
import json
import torch
from torch import nn
import numpy as np
import random
from typing import Optional
import math

class DDPMSampler:
    def __init__(self, noise_step: int=1000, beta_start: float=0.00085, beta_end: float=0.0120, use_cosine_schedule: bool=False, device: str='cpu'):
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, noise_step, device=device) ** 2
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.noise_step = noise_step
        
        self.timesteps = torch.from_numpy(np.arange(0, noise_step)[::-1].copy()).to(device)   

        # Cosine-based noise schedule
        if use_cosine_schedule:
            s = 0.008
            f_t = lambda t: np.cos((t/noise_step + s)/(1 + s) * np.pi/2) ** 2
            self.alphas_hat = f_t(torch.arange(0, noise_step + 1)) / f_t(0)
            self.alphas_hat = self.alphas_hat.to(device)
            self.betas = torch.clip(1 - self.alphas_hat[1:]/self.alphas_hat[:-1], 0, 0.999)
            self.alphas = torch.clip(1. - self.betas, 0, 0.999)
            self.alphas_hat = torch.clip(self.alphas_hat[1:], 0, 0.999)

    def _set_inference_steps(self, inference_steps=50):
        self.inference_steps = inference_steps
        step = self.noise_step // self.inference_steps
        self.timesteps = torch.from_numpy((np.arange(0, self.inference_steps) * step).round()[::-1].copy().astype(np.int64))
        

    def _sample_timestep(self, n):
        return torch.randint(low=0, high=self.noise_step, size=(n,))
        
    def _get_prev_timestep(self, timestep: int):
        prev_t = timestep - self.noise_step // self.inference_steps
        return prev_t

    def set_strength(self, strength: float=0.8):
        start_t = self.inference_steps - int(self.inference_steps * strength)
        self.timesteps = self.timesteps[start_t:]

    # x_t ~ q(x_t | x_0) = N(x_t, sqrt(a_hat_t) * x_0, sqrt(1 - a_hat_t) * I)
    def forward_process(self, x_0: torch.Tensor, timestep: torch.LongTensor, noise: Optional[torch.Tensor] = None):
        # x_0: (b, c, h, w)
        t = timestep
        # (n,) -> (n, 1, 1, 1)
        alpha_hat_t = self.alphas_hat.to(x_0.device)[t][:, None, None, None]
        
        if noise is None:
            noise = torch.randn(x_0.shape, dtype=x_0.dtype, device=x_0.device)
            
        latent = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise
        return latent, noise

    # x_(t-1) ~ p(x_(t-1) | x_t) = N(x_(t - 1), mu_theta(x_t, x_0), beta_tilda_t * I)
    # mu_theta(x_t, x-0)1/sqrt(alpha_hat_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_hat_t) * epsilon_t)
    # beta_tilda_t = (1 - alpha_hat_(t-1)) / (1 - alpha_t) * beta_t
    def reverse_process(self, x_t: torch.Tensor, timestep: int, model_output=torch.Tensor) -> torch.Tensor:
        t = timestep
        prev_t = self._get_prev_timestep(t)
        alpha_t = self.alphas[t]
        alpha_hat_t = self.alphas_hat[t]
        prev_alpha_hat_t = self.alphas_hat[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=x_t.device)

        current_alpha_t = torch.clip(alpha_hat_t / prev_alpha_hat_t, 0, 0.999)
        current_beta_t = 1 - current_alpha_t

        mu = 1/torch.sqrt(current_alpha_t) * (x_t - ((1 - current_alpha_t)/torch.sqrt(1 - alpha_hat_t)) * model_output)
        
        stdev = 0
        if t > 0:
            variance = (1 - prev_alpha_hat_t) / (1 - alpha_hat_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            stdev = torch.sqrt(variance)
            
        noise = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device)
        less_noise_sample = mu + stdev * noise
        return less_noise_sample
    
    @staticmethod
    def from_config(cfg_path: str, use_cosine_schedule: bool=False, device: str='cpu'):
        with open(os.path.join(cfg_path, "scheduler_config.json"), 'r') as f:
            config = json.load(f)
        scheduler = DDPMSampler(noise_step=config["num_train_timesteps"], beta_start=config["beta_start"], beta_end=config["beta_end"], use_cosine_schedule=use_cosine_schedule, device=device, prediction_type=config["prediction_type"])
        return scheduler
        