import os
import json

import torch
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union
from dataclasses import dataclass, field

from .attention import MultiheadSelfAttention
from ..activation_fn import GeGLU
from utils.model_converter import load_unet_weights_v2_1, load_unet_weights_v1_5

@dataclass
class UNetConfig:
    act_fn: str = "silu"
    attention_head_dim: list = field(default_factory=[5, 10, 20, 20])
    block_out_channels: list = field(default_factory=[320, 640, 1280, 1280])
    down_block_types: list = field(default_factory=["CrossAttnDownBlock2D",
                                                    "CrossAttnDownBlock2D",
                                                    "CrossAttnDownBlock2D",
                                                    "DownBlock2D"])
    center_input_sample: bool = False
    cross_attention_dim: int = 1024
    downsample_padding: int = 1
    dual_cross_attention: bool = False
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    in_channels: int = 4
    layers_per_block: int = 2
    mid_block_scale_factor: int = 1
    norm_eps: float = 1e-05
    norm_num_groups: int = 32
    num_class_embeds: int = None
    only_cross_attention: bool = False
    out_channels: int = 4
    sample_size: int = 96
    use_linear_projection: bool = True
    upcast_attention: bool = True
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            act_fn=data["act_fn"],
            attention_head_dim=data["attention_head_dim"],
            block_out_channels=data["block_out_channels"],
            center_input_sample=data["center_input_sample"],
            cross_attention_dim=data["cross_attention_dim"],
            downsample_padding=data["downsample_padding"],
            flip_sin_to_cos=data["flip_sin_to_cos"],
            freq_shift=data["freq_shift"],
            in_channels=data["in_channels"],
            layers_per_block=data["layers_per_block"],
            mid_block_scale_factor=data["mid_block_scale_factor"],
            norm_eps=data["norm_eps"],
            norm_num_groups=data["norm_num_groups"],
            out_channels=data["out_channels"],
            sample_size=data["sample_size"],
            down_block_types=data["down_block_types"]
        )
class UNet_TransformerEncoder(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, cond_dim: int, use_lora: bool, eps: float=1e-05):
        super().__init__()
        channels = embedding_dim * num_heads
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.transformer_block = UNet_AttentionBlock(num_heads=num_heads, embedding_dim=channels, cond_dim=cond_dim, use_lora=use_lora)
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor=None) -> torch.Tensor: 
        # x: (b, c, h, w)
        b, c, h, w = x.shape
        
        x_in = x

        x = self.groupnorm(x)
        x = self.conv_input(x)
        # (b, c, h, w) -> (b, c, h * w) -> (b, h * w, c)
        x = x.view(b, c, -1).transpose(-1, -2)
        x = self.transformer_block(x=x, cond=cond)
        
        x = x.transpose(-1, -2).view(b, c, h, w)

        x = self.conv_output(x)
        
        x += x_in
        
        return x
        
class UNet_AttentionBlock(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, cond_dim: int, use_lora: bool=False, eps: float=1e-05):
        super().__init__()
        
        if embedding_dim % num_heads:
            raise ValueError('Number of heads must be divisible by Embedding Dimension')
            
        self.head_dim = embedding_dim // num_heads

        self.layernorm_1 = nn.LayerNorm(embedding_dim, eps=eps)
        self.attn1 = MultiheadSelfAttention(num_heads=num_heads, embedding_dim=embedding_dim, cond_dim=None, qkv_bias=False)
        
        self.layernorm_2 = nn.LayerNorm(embedding_dim, eps=eps)
        self.attn2 = MultiheadSelfAttention(num_heads=num_heads, embedding_dim=embedding_dim, cond_dim=cond_dim, qkv_bias=False)

        self.layernorm_3 = nn.LayerNorm(embedding_dim, eps=eps)
                
        self.ffn = nn.Sequential(
            GeGLU(embedding_dim, embedding_dim * 4),
            nn.Linear(embedding_dim * 4, embedding_dim))

        if use_lora:
            self.attn1.proj_q.parametrizations.weight[0].enabled = True
            self.attn1.proj_k.parametrizations.weight[0].enabled = True
            self.attn1.proj_v.parametrizations.weight[0].enabled = True
            self.attn1.proj_out.parametrizations.weight[0].enabled = True
            
            self.attn2.proj_q.parametrizations.weight[0].enabled = True
            self.attn2.proj_k.parametrizations.weight[0].enabled = True
            self.attn2.proj_v.parametrizations.weight[0].enabled = True
            self.attn2.proj_out.parametrizations.weight[0].enabled = True
        
        self.gradient_checkpointing = False
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual_x = x
        x = self.layernorm_1(x)
        if self.gradient_checkpointing:
            x = checkpoint.checkpoint(self.attn1, x, use_reentrant=False)
        else:
            x = self.attn1(x)
        x += residual_x
        
        residual_x = x
        x = self.layernorm_2(x)
        if self.gradient_checkpointing:
            x = checkpoint.checkpoint(self.attn2, x, cond, use_reentrant=False)
        else:
            x = self.attn2(x, cond=cond)
        x += residual_x
        
        residual_x = x
        x = self.layernorm_3(x)
        x = self.ffn(x)

        x += residual_x
        
        return x
        

class UNet_ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_embed_dim: int, eps: float=1e-05):
        super().__init__()
        
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        
        self.t_embed = nn.Linear(t_embed_dim, out_channels)
        
        if in_channels == out_channels:
            self.proj_input = nn.Identity()
        else:
            self.proj_input = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

        self.silu_1 = nn.SiLU()
        self.silu_2 = nn.SiLU()
        self.silu_t_embed = nn.SiLU()

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        # x: (n, c, h, w)
        h = self.groupnorm_1(x)
        h = self.silu_1(h)
        h = self.conv_1(h)
        
        
        # time: (1, t_embed_dim) -> (1, out_channels)
        time = self.silu_t_embed(t_embed)
        time = self.t_embed(time)[:, :, None, None]
        
        # (n, out_channels, h, w) + (1, out_channels, 1, 1) -> (n, out_channels, h, w)
        h += time
        
        h = self.groupnorm_2(h)
        h = self.silu_2(h)
        h = self.conv_2(h)
        
        x = self.proj_input(x)     
        h += x
        
        return h

class TimeEmbedding(nn.Module):
    def __init__(self, t_embed_dim: int=320):
        super().__init__()
        
        self.t_embed_dim = t_embed_dim
        self.ffn = nn.Sequential(
            # (1, 320) -> (1, 1280)
            nn.Linear(t_embed_dim, t_embed_dim * 4),
            nn.SiLU(),
            # (1, 1280) -> (1, 1280)
            nn.Linear(t_embed_dim * 4,  t_embed_dim * 4))

    def _get_time_embedding(self, timestep: torch.Tensor):
        import math
        half = self.t_embed_dim // 2
        freqs = -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timestep.device)
        freqs = freqs / half
        freqs = torch.exp(freqs)
        x = timestep[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
            
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        t_embed = self._get_time_embedding(timestep)
        return self.ffn(t_embed)

class SwitcSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, cond=None) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNet_ResBlock):
                x = layer(x, t_embed)
            elif isinstance(layer, UNet_TransformerEncoder):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x
        
class UNet_Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)    
        return x

class UNet_Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, upscale=True) -> torch.Tensor:
        if upscale:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class UNet_Encoder(nn.Module):
    def __init__(self, in_channels: int=4, num_attention_heads: List[int]=[5, 10, 20, 20], t_embed_dim: int=1280, cross_attention_dim: List[int]=[1024, 1024, 1024, 1024], block_out_channels: List[int]=[320, 640, 1280, 1280], use_lora: bool=False, eps: float=1e-05):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        
        self.down = nn.ModuleList()
        
        block_in_channels = [block_out_channels[0]] + block_out_channels
        
        for i in range(len(block_out_channels)):
            down = nn.Module()
            in_channels = block_in_channels[i]
            out_channels = block_out_channels[i]
            
            if i != len(block_out_channels) - 1:
                block = nn.Sequential(
                    SwitcSequential(UNet_ResBlock(in_channels, out_channels, t_embed_dim), UNet_TransformerEncoder(num_heads=num_attention_heads[i], embedding_dim=out_channels//num_attention_heads[i], cond_dim=cross_attention_dim[i], use_lora=use_lora, eps=eps)),
                    SwitcSequential(UNet_ResBlock(out_channels, out_channels, t_embed_dim), UNet_TransformerEncoder(num_heads=num_attention_heads[i], embedding_dim=out_channels//num_attention_heads[i], cond_dim=cross_attention_dim[i], use_lora=use_lora, eps=eps)),)
                downsample = UNet_Downsample(out_channels)
            else:
                block = nn.Sequential(
                SwitcSequential(UNet_ResBlock(in_channels, out_channels, t_embed_dim, eps=eps)),
                SwitcSequential(UNet_ResBlock(out_channels, out_channels, t_embed_dim, eps=eps)))
                
                downsample = nn.Identity()
            
            down.block = block
            down.downsample = downsample
            
            self.down.append(down)
       
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.conv_in(x)
        skip_connections = [x]
        for down in self.down:
            for layer in down.block:
                x = layer(x, t_embed, cond)
                skip_connections.append(x)
            
            x = down.downsample(x)
            if not isinstance(down.downsample, nn.Identity):
                skip_connections.append(x)
        return x, skip_connections

class UNet_Decoder(nn.Module):
    def __init__(self, num_attention_heads: List[int]=[5, 10, 20, 20], t_embed_dim: int=1280, cross_attention_dim: List[int]=[1024, 1024, 1024, 1024], block_out_channels: List[int]=[320, 640, 1280, 1280], use_lora: bool=False, eps: float=1e-05):
        super().__init__()
        ch = 320
        block_in_channels = block_out_channels + [block_out_channels[-1]]
        self.up = nn.ModuleList()
        for i in reversed(range(len(block_out_channels))):
            up = nn.Module()
            in_ch = block_in_channels[i + 1]
            out_ch = block_out_channels[i]
            
            if i > 0:
                mid_ch = block_in_channels[i-1]
            else:
                mid_ch = ch
                
            if i == len(block_out_channels) - 1:
                block = nn.Sequential(
                    SwitcSequential(UNet_ResBlock(in_ch + out_ch, out_ch, t_embed_dim, eps=eps)),
                    SwitcSequential(UNet_ResBlock(out_ch + out_ch, out_ch, t_embed_dim, eps=eps)),
                    SwitcSequential(UNet_ResBlock(out_ch + mid_ch, out_ch, t_embed_dim, eps=eps)))
            else:
                block = nn.Sequential(
                    SwitcSequential(UNet_ResBlock(in_ch + out_ch, out_ch, t_embed_dim, eps=eps), UNet_TransformerEncoder(num_heads=num_attention_heads[i], embedding_dim=out_ch//num_attention_heads[i], cond_dim=cross_attention_dim[i], use_lora=use_lora, eps=eps)), 
                    SwitcSequential(UNet_ResBlock(out_ch + out_ch, out_ch, t_embed_dim, eps=eps), UNet_TransformerEncoder(num_heads=num_attention_heads[i], embedding_dim=out_ch//num_attention_heads[i], cond_dim=cross_attention_dim[i], use_lora=use_lora, eps=eps)),
                    SwitcSequential(UNet_ResBlock(out_ch + mid_ch, out_ch, t_embed_dim, eps=eps), UNet_TransformerEncoder(num_heads=num_attention_heads[i], embedding_dim=out_ch//num_attention_heads[i], cond_dim=cross_attention_dim[i], use_lora=use_lora, eps=eps)))
                   
            
            if i != 0:
                upsample = UNet_Upsample(out_ch)
            else:
                upsample = nn.Identity()

            up.block = block
            up.upsample = upsample

            self.up.append(up)

            

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor], t_embed: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (b, c, h, w)
        for i, up in enumerate(self.up):
            prev_hw = skip_connections[-1].shape[-1]
            for layer in up.block:
                tmp = skip_connections.pop()
                x = torch.cat([x, tmp], dim=1)
                x = layer(x, t_embed, cond)
            
            if skip_connections and skip_connections[-1].shape[-1] == prev_hw:
                x = up.upsample(x, upscale=False)
            else:
                x = up.upsample(x)
              
        return x

class UNet(nn.Module):
    def __init__(self, 
                 attention_head_dim: Union[int|List[int]] = 8,
                 cross_attention_dim: Union[int|List[int]] = 768,
                 in_channels: int = 4,
                 out_channels: int = 4,
                 block_out_channels: List[int] = [320,640,1280,1280],
                 down_block_types: List[int] = ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
                 t_embed_dim: int = 320, 
                 use_lora=False,
                 num_attention_heads: Optional[Union[int|List[int]]]=None,
                 eps: float=1e-05):
        super().__init__()
        if isinstance(attention_head_dim, int):
            attention_head_dim = [attention_head_dim] * len(down_block_types)
        
        if isinstance(cross_attention_dim, int):
            cross_attention_dim = [cross_attention_dim] * len(down_block_types)
        
        if num_attention_heads is None:
            num_attention_heads = attention_head_dim
        
        self.time_embedding = TimeEmbedding(t_embed_dim)
        self.encoder = UNet_Encoder(in_channels=in_channels, 
                                    num_attention_heads=num_attention_heads, 
                                    t_embed_dim=t_embed_dim * 4, 
                                    cross_attention_dim=cross_attention_dim,
                                    block_out_channels=block_out_channels,
                                    use_lora=use_lora,
                                    eps=eps)
        self.bottleneck = SwitcSequential(
            UNet_ResBlock(1280, 1280, t_embed_dim * 4),
            UNet_TransformerEncoder(num_heads=num_attention_heads[-1], 
                                    embedding_dim=block_out_channels[-1]//num_attention_heads[-1], 
                                    cond_dim=cross_attention_dim[-1], 
                                    use_lora=use_lora,
                                    eps=eps),
            UNet_ResBlock(1280, 1280, t_embed_dim * 4)
        )
        self.decoder = UNet_Decoder(num_attention_heads=num_attention_heads, 
                                    t_embed_dim=t_embed_dim * 4, 
                                    cross_attention_dim=cross_attention_dim, 
                                    block_out_channels=block_out_channels,
                                    use_lora=use_lora,
                                    eps=eps)
        self.output = nn.Sequential(
            nn.GroupNorm(32, 320, eps=eps, affine=True),
            nn.SiLU(),
            nn.Conv2d(320, out_channels, kernel_size=3, stride=1, padding=1))
        
        
    def gradient_checkpointing_enabled(self, enabled=False):
        for name, module in self.encoder.named_modules():
            if isinstance(module, UNet_AttentionBlock):
                module.gradient_checkpointing = enabled
                
        for name, module in self.bottleneck.named_modules():
            if isinstance(module, UNet_AttentionBlock):
                module.gradient_checkpointing = enabled
                
        for name, module in self.decoder.named_modules():
            if isinstance(module, UNet_AttentionBlock):
                module.gradient_checkpointing = enabled
    
    def enable_flash_attn(self):
        for name, module in self.encoder.named_modules():
            if isinstance(module, MultiheadSelfAttention):
                module.use_flash_attention = True
                
        for name, module in self.bottleneck.named_modules():
            if isinstance(module, MultiheadSelfAttention):
                module.use_flash_attention = True
                
        for name, module in self.decoder.named_modules():
            if isinstance(module, MultiheadSelfAttention):
                module.use_flash_attention = True
        
                
    def forward(self, x: torch.Tensor, timestep: torch.LongTensor, cond: torch.Tensor) -> torch.Tensor:
        # t: (n,) -> (n, 1280)
        t_embed = self.time_embedding(timestep)
        x, skip_connections = self.encoder(x, t_embed, cond)
        
        x = self.bottleneck(x, t_embed, cond)
        
        x = self.decoder(x, skip_connections, t_embed, cond)
        # torch.save(x, "hidden_states1.pt")
        # import sys
        # sys.exit()
        output = self.output(x)
        return output
    
    @staticmethod
    def from_pretrained(pretrained_dir: str, device: str='cpu', sd_version: str = "1.5"):
        if sd_version == "1.5":
            state_dict = load_unet_weights_v1_5(os.path.join(pretrained_dir, "diffusion_pytorch_model.safetensors"), device=device)['unet']
        else:
            state_dict = load_unet_weights_v2_1(os.path.join(pretrained_dir, "diffusion_pytorch_model.safetensors"), device=device)['unet']
        with open(os.path.join(pretrained_dir, "config.json"), "r") as f:
            cfg = UNetConfig.from_dict(json.load(f))
            model = UNet(attention_head_dim=cfg.attention_head_dim,
                         cross_attention_dim=cfg.cross_attention_dim,
                         in_channels=cfg.in_channels,
                         out_channels=cfg.out_channels,
                         block_out_channels=cfg.block_out_channels,
                         down_block_types=cfg.down_block_types,
                         eps=cfg.norm_eps)
        model.load_state_dict(state_dict=state_dict, strict=True)
        return model