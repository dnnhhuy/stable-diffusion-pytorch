import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from typing import List

class LoraLayer(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alphas=1):
        super().__init__()
        self.alphas = alphas
        self.scale = rank / self.alphas
        self.enabled = False
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)))
        nn.init.normal_(self.lora_A)
    
    def forward(self, W_0):
        if self.enabled:
            return W_0 + torch.matmul(self.lora_B, self.lora_A) * self.scale
        else:
            return W_0

def parametrize_linear_layer(layer, rank, alphas):
    features_in, features_out = layer.weight.shape
    return LoraLayer(features_in, features_out, rank, alphas)

def enable_lora(model: nn.Module, lora_modules: List[str], enabled=False):
    for name, module in model.named_modules():
       if name.split('.')[-1] in lora_modules:
           module.parametrizations.weight[0].enabled = True
    return model

def get_lora_model(model: nn.Module, rank: float, alphas: float, lora_modules=List[str]):
    for name, module in model.named_modules():
       if name.split('.')[-1] in lora_modules:
           parametrize.register_parametrization(module, "weight", parametrization=parametrize_linear_layer(module, rank=rank, alphas=alphas))
           module.parametrizations.weight[0].enabled = True
           
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    return model