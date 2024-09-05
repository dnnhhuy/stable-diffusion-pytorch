import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

class LoraLayer(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alphas=1):
        super().__init__()
        self.scale = rank / alphas
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


def get_lora_model(model: nn.Module):
    for param in model.cond_encoder.parameters():
        param.requires_grad = False
    
    for param in model.vae.parameters():
        param.requires_grad = False
        
    peft_module = ['proj_q', 'proj_k', 'proj_v', 'proj_out']
    for name, module in model.unet.named_modules():
       if name.split('.')[-1] in peft_module:
           parametrize.register_parametrization(module, "weight", parametrization=parametrize_linear_layer(module, rank=8, alphas=8))
           module.parametrizations.weight[0].enabled = True
           
    for name, param in model.unet.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    
    return model