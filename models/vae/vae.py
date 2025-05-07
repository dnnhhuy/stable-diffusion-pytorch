import os 
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from .resnet import ResidualBlock

from typing import List
from safetensors.torch import load_file

from utils.model_converter import convert_swiftbrush_vae

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
    

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.pad = (0, 1, 0, 1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.pad)
        x = self.conv(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x
        
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, eps: float=1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps)
        self.query = nn.Linear(in_channels, in_channels, bias=True)
        self.key = nn.Linear(in_channels, in_channels, bias=True)
        self.value = nn.Linear(in_channels, in_channels, bias=True)
        
        self.num_heads = 1
        self.head_dim = in_channels // self.num_heads
        self.proj_attn = nn.Linear(in_channels, in_channels, bias=True)
        self.use_flash_attention = False
        self.dropout = 0.0
    
    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lookahead_mask: bool):
        batch_size, seq_len, embedding_dim = q.shape
        # (batch_size, seq_len, embedding_dim) -> (n, seq_len, num_heads, head_dim) -> (n, num_heads, seq_len, head_dim)
        q = q.view(*q.shape[:2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # (n, num_heads, seq_len, head_dim) @ (n, num_heads, head_dim, seq_len) -> (n, num_heads, seq_len, seq_len)
        attn_weights = q @ k.transpose(-1, -2)
        
        attn_weights = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout,
            is_causal=lookahead_mask
        )
        
        # (n, num_heads, seq_len, head_dim) -> (n, seq_len, num_heads, head_dim) -> (n, seq_len, embedding_dim)
        attn_weights = attn_weights.transpose(1, 2).reshape((batch_size, seq_len, embedding_dim))
        
        attn_weights = F.dropout(attn_weights, self.dropout)

        out = self.proj_attn(attn_weights)
        return out
    
    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lookahead_mask: bool):
        batch_size, seq_len, embedding_dim = q.shape
        
        q = q.view(*q.shape[:2], self.num_heads, self.head_dim)
        k = k.view(*k.shape[:2], self.num_heads, self.head_dim)
        v = v.view(*v.shape[:2], self.num_heads, self.head_dim)
        
        out = flash_attn_func(q, k, v, dropout_p=self.dropout, softmax_scale=self.head_dim ** -0.5, causal=lookahead_mask)
        
        out = out.reshape((batch_size, seq_len, embedding_dim))
        
        out = F.dropout(out, self.dropout)
        
        out = self.proj_attn(out)
        
        return out
        
        
    def attn(self, x: torch.Tensor, cond: torch.Tensor=None, lookahead_mask: bool=False) -> torch.Tensor:
        # x: (n, seq_len, embedding_dim)
        # cond: (n, seq_len, cond_dim)
        if cond is None:
            cond = x
            
        if len(cond.shape) < len(x.shape):
            cond = cond.unsqueeze(1)
        
        q = self.query(x)
        k = self.key(cond)
        v = self.value(cond)
        
        if self.use_flash_attention and self.head_dim <= 128 and flash_attn_func is not None:
            return self.flash_attention(q, k, v, lookahead_mask)
        
        else:
            return self.normal_attention(q, k, v, lookahead_mask)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, c, h, w)
        batch_size, channels, h, w = x.shape
        x_norm = self.group_norm(x)
        # (n, c, h, w) -> (n, c, h * w) -> (n, h * w, c)
        x_norm = x_norm.view((batch_size, channels, -1)).transpose(1, 2)
        
        # (n, h * w, c)
        out = self.attn(x=x_norm)

        # (n, h * w, c) -> (n, c, h * w) -> (n, c, h, w)
        out = out.transpose(1, 2).view(x.shape)
        return out + x
        
class VAE_Encoder(nn.Module):
    def __init__(self, in_channels: int, ch_mult: List[int]=[1, 2, 4, 4], dropout: float=0.0, z_channels: int=4, eps: float=1e-6):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)

        # start
        self.down_blocks = nn.ModuleList()
        in_ch_mult = [1] + ch_mult
        ch = 128
        for i in range(len(ch_mult)):
            block_in = ch * in_ch_mult[i]
            block_out = ch * ch_mult[i]
            resnets = nn.Sequential( 
                ResidualBlock(block_in, block_out, dropout),
                ResidualBlock(block_out, block_out, dropout),
            )
            
            down = nn.Module()
            down.resnets = resnets
            if i != len(ch_mult) - 1:
                down.downsamplers = nn.Sequential(Downsample(block_out))
            else:
                down.downsamplers = nn.Identity()
                
            self.down_blocks.append(down)
            curr_channels = block_out

        # middle
        self.mid_block = nn.Module()
        self.mid_block.resnets = nn.ModuleList([ResidualBlock(curr_channels, curr_channels),
                                                ResidualBlock(curr_channels, curr_channels)])
        self.mid_block.attentions = nn.ModuleList([AttentionBlock(in_channels=curr_channels)])
        
        # end
        self.conv_norm_out = nn.GroupNorm(num_groups=32, num_channels=curr_channels, eps=eps)
        self.conv_out = nn.Conv2d(curr_channels, 2*z_channels, kernel_size=3, stride=1, padding=1)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        
        for down in self.down_blocks:
            x = down.resnets(x)
            x = down.downsamplers(x)
        
        x = self.mid_block.resnets[0](x)
        x = self.mid_block.attentions[0](x)
        x = self.mid_block.resnets[1](x)
        x = self.conv_norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class VAE_Decoder(nn.Module):
    def __init__(self, ch_mult: List[int]=[1, 2, 4, 4], dropout: float=0.0, z_channels: int=4, out_channels: int=3):
        super().__init__()

        ch = 128
        block_in = ch*ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
    
        # mid
        self.mid_block = nn.Module()
        self.mid_block.attentions = nn.ModuleList([AttentionBlock(in_channels=block_in)])
        self.mid_block.resnets = nn.ModuleList([ResidualBlock(block_in, block_in),
                                                 ResidualBlock(block_in, block_in)])

        # upsampling
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[i]
            up = nn.Module()
            up.resnets = nn.Sequential(
                ResidualBlock(block_in, block_out),
                ResidualBlock(block_out, block_out),
                ResidualBlock(block_out, block_out)
            )
            if i != 0:
                up.upsamplers = nn.Sequential(UpSample(in_channels=block_out))
            else:
                up.upsamplers = nn.Sequential(nn.Identity())
                
            self.up_blocks.append(up)
            block_in = block_out

        self.conv_norm_out = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, stride=1, padding=1)

            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, h, w)
        x = self.conv_in(x)

        x = self.mid_block.resnets[0](x)
        x = self.mid_block.attentions[0](x)
        x = self.mid_block.resnets[1](x)

        for up in self.up_blocks:
            x = up.resnets(x)
            x = up.upsamplers(x)
        x = self.conv_norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x
        
        
class VAE(nn.Module):
    def __init__(self, in_channels: int=3, z_channels: int=4):
        super().__init__()
        self.encoder = VAE_Encoder(in_channels=in_channels)
        self.decoder = VAE_Decoder(z_channels=z_channels)
        
        self.quant_conv = nn.Conv2d(2*z_channels, 2*z_channels, kernel_size=1, stride=1, padding=0)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1, padding=0)

    def encode(self, x: torch.Tensor, noise=None, generator=None) -> torch.Tensor:
        # z: (n, c, h, w)
        z = self.encoder(x)
        z = self.quant_conv(z)
        mean, log_variance = z.chunk(2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        
        variance = torch.exp(log_variance)
        stdev = torch.exp(0.5 * log_variance)
        
        if noise is not None:
            output = mean + stdev * noise
        else:
            sample = torch.randn(stdev.shape, generator=generator, device=stdev.device)
            output = mean + stdev * sample
            output = output * 0.18215
        return output, mean, stdev
        

    def decode(self, z: torch.Tensor):
        z = z / 0.18215
        z = self.post_quant_conv(z)
        out = self.decoder(z) 
        return out
    
    @staticmethod
    def from_pretrained(pretrained_path: str, device: str='cpu'):
        with open(os.path.join(pretrained_path, "config.json"), "r") as f:
            cfg_dict = json.load(f)
            model = VAE(in_channels=cfg_dict['in_channels'], z_channels=cfg_dict['latent_channels'])
        try:
            state_dict = load_file(os.path.join(pretrained_path, 'diffusion_pytorch_model.safetensors'), device=device)
            model.load_state_dict(state_dict, strict=True)
        except:
            state_dict = convert_swiftbrush_vae(os.path.join(pretrained_path, 'diffusion_pytorch_model.safetensors'), device=device)
            model.load_state_dict(state_dict, strict=True)
        return model
            
        

class VQVAE(nn.Module):
    def __init__(self, codebook_size: int=1024, in_channels: int=3, z_channels: int=4, use_ema: bool=False, beta: float=0.995):
        super().__init__()
        self.encoder = VAE_Encoder(in_channels=in_channels, z_channels=z_channels)
        self.decoder = VAE_Decoder(z_channels=z_channels*2, out_channels=in_channels)
        
        self.codebook_size = codebook_size
        self.codebook_dim = z_channels * 2
        
        # (codebook_size, z_channels)
        self.quant_embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        
        self.use_ema = use_ema

        if self.use_ema:
            self.beta = beta
            self.register_buffer("M", torch.zeros(codebook_size))
            self.M = nn.Parameter(torch.Tensor(self.codebook_size, self.codebook_dim))
            self.M.data.normal_()

    def encode(self, x: torch.Tensor, is_training: bool=False) -> torch.Tensor:
        # z: (n, c, h, w)
        z = self.encoder(x)
        n, c, h, w = z.shape

        # (n, c, h, w) -> (n, c, h * w) -> (n, h * w, c)
        z = z.view(n, c, -1).permute(0, 2, 1)

        # distance -> (n, h * w, codebook_size)
        distance = torch.cdist(z, self.quant_embedding.weight.unsqueeze(0).repeat((z.shape[0], 1, 1)))

        # min_indices -> (n, h * w)
        min_indices = torch.argmin(distance, dim=-1)

        # quant_out -> (n*h*w, c)
        quant_out = torch.index_select(self.quant_embedding.weight, 0, min_indices.view(-1))

        if self.use_ema:
            if is_training:
                self.update_quant_embedding(min_indices, z)
            
             # (n, h*w, c) -> (n*h*w, c)
            z = z.reshape((-1, z.size(-1)))
            
            commitment_loss = F.mse_loss(z, quant_out.detach())
            quantize_loss = commitment_loss

            # Copy gradient
            quant_out = z + (quant_out - z).detach()
    
            quant_out = quant_out.reshape((n, h, w, c)).permute(0, 3, 1, 2)
            return quant_out, quantize_loss, min_indices
            
        else:
            # (n, h*w, c) -> (n*h*w, c)
            z = z.reshape((-1, z.size(-1)))
            
            vq_loss = F.mse_loss(z.detach(), quant_out)
            commitment_loss = F.mse_loss(z, quant_out.detach())
            
            quantize_loss =  vq_loss + commitment_loss
            # Copy gradient
            quant_out = z + (quant_out - z).detach()
    
            quant_out = quant_out.reshape((n, h, w, c)).permute(0, 3, 1, 2)
            min_indices = min_indices.reshape((-1, quant_out.shape[-2], quant_out.shape[-1]))
            return quant_out, quantize_loss, min_indices

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z) 
        return out

    def update_quant_embedding(self, min_indices, encoder_input):
        # encodings -> (n * h * w, codebook_size)
        encodings = torch.zeros(min_indices.view(-1).size(0), self.codebook_size)
        encodings.scatter_(1, min_indices.view(-1)[:, None], 1)

        # (codebook_size)
        self.N = self.beta * self.N + (1 - self.beta) * torch.sum(encodings, 0)

        # (n, h * w, c) -> (n * h * w, c)
        encoder_input = encoder_input.view(-1, self.codebook_dim)

        # (codebook_size, c)
        self.M = nn.Parameter(self.beta * self.M + (1 - self.beta) * (encodings.T @ encoder_input), requires_grad=False)

        self.quant_embedding.weight = nn.Parameter(self.M / self.N.unsqueeze(1))

        
        

        
        
        
                