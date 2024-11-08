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
        self.lora_A = nn.Parameter(torch.zeros((features_out, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, features_in)))
        nn.init.normal_(self.lora_A)
    
    def forward(self, W_0):
        if self.enabled:
            return W_0 + torch.matmul(self.lora_A, self.lora_B) * self.scale
        else:
            return W_0
    
class LoraConvLayer(nn.Module):
    def __init__(self, features_in, features_out, kernel_size, rank=1, alphas=1):
        super().__init__()
        self.alphas = alphas
        self.scale = rank / self.alphas
        self.enabled = False
        self.lora_A = nn.Parameter(torch.zeros((features_out, rank, kernel_size, kernel_size)))
        self.lora_B = nn.Parameter(torch.zeros((rank, features_in, kernel_size, kernel_size)))
        nn.init.normal_(self.lora_A)
    
    def forward(self, W_0):
        if self.enabled:
            return W_0 + torch.einsum("abcd,becd->aecd", self.lora_A, self.lora_B) * self.scale
        else:
            return W_0

def parametrize_linear_layer(layer, rank, alphas):
    features_out, features_in = layer.weight.shape
    return LoraLayer(features_in, features_out, rank, alphas)

def parametrize_conv_layer(layer, rank, alphas):
    features_out, features_in = layer.weight.shape[0], layer.weight.shape[1]
    kernel_size = layer.weight.shape[2]
    return LoraConvLayer(features_in=features_in, features_out=features_out, kernel_size=kernel_size, rank=rank, alphas=alphas)

def enable_lora(model: nn.Module, lora_modules: List[str], enabled=False):
    for name, module in model.named_modules():
       for lora_module in lora_modules:
            if name.endswith(lora_module):
                module.parametrizations.weight[0].enabled = enabled
    return model

def get_lora_model(model: nn.Module, rank: float, alphas: float, lora_modules=List[str]):
    for name, module in model.named_modules():
        for lora_module in lora_modules:
            if name.endswith(lora_module):
                if isinstance(module, nn.Linear):
                    parametrize.register_parametrization(module, "weight", parametrization=parametrize_linear_layer(module, rank=rank, alphas=alphas))
                    module.parametrizations.weight[0].enabled = True
                elif isinstance(module, nn.Conv2d):
                    parametrize.register_parametrization(module, "weight", parametrization=parametrize_conv_layer(module, rank=rank, alphas=alphas))
                    module.parametrizations.weight[0].enabled = True
                    
           
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    return model