import torch
from torch import nn
from torch.nn import functional as F
from .attention import MultiheadSelfAttention
from .activation_fn import GeGELU
from typing import Optional, List

class UNet_TransformerEncoder(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, cond_dim: int=768):
        super().__init__()
        channels = embedding_dim * num_heads
        self.groupnorm = nn.GroupNorm(32, channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.transformer_block = UNet_AttentionBlock(num_heads=num_heads, embedding_dim=channels, cond_dim=cond_dim)

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

        return x + x_in
        
class UNet_AttentionBlock(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, cond_dim: int=768):
        super().__init__()
        
        if embedding_dim % num_heads:
            raise ValueError('Number of heads must be divisible by Embedding Dimension')
            
        self.head_dim = embedding_dim // num_heads

        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.attn1 = MultiheadSelfAttention(num_heads=num_heads, embedding_dim=embedding_dim, cond_dim=None, qkv_bias=False)
        
        self.layernorm_2 = nn.LayerNorm(embedding_dim)
        self.attn2 = MultiheadSelfAttention(num_heads=num_heads, embedding_dim=embedding_dim, cond_dim=cond_dim, qkv_bias=False)

        self.layernorm_3 = nn.LayerNorm(embedding_dim)
                
        self.ffn = nn.Sequential(
            GeGELU(embedding_dim, embedding_dim * 4),
            nn.Linear(embedding_dim * 4, embedding_dim))
        

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.attn1(self.layernorm_1(x)) + x

        x = self.attn2(self.layernorm_2(x), cond=cond) + x

        x = self.ffn(self.layernorm_3(x)) + x
        
        return x
        

class UNet_ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_embed_dim: int):
            super().__init__()
            
            self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
            self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

            self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
            self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            
            self.t_embed = nn.Linear(t_embed_dim, out_channels)
            
            if in_channels == out_channels:
                self.proj_input = nn.Identity()
            else:
                self.proj_input = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        # x: (n, c, h, w)
        h = self.groupnorm_1(x)
        h = F.silu(h)
        h = self.conv_1(h)

        # time: (1, t_embed_dim) -> (1, out_channels)
        time = F.silu(t_embed)
        time = self.t_embed(time)
        # (n, out_channels, h, w) + (1, out_channels, 1, 1) -> (n, out_channels, h, w)
        h = h + time[:, :, None, None]

        h = self.groupnorm_2(h)
        h = F.silu(h)
        h = self.conv_2(h)
        return h + self.proj_input(x)

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

    def _get_time_embedding(self, timestep):
        half = self.t_embed_dim // 2
        freqs = torch.pow(10000, -torch.arange(0, half, dtype=torch.float32)/half)
        x = torch.tensor([timestep], dtype=torch.float32, device=timestep.device)[None, :] * freqs[None, :].to(timestep.device)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
            
    def forward(self, timestep: int) -> torch.Tensor:
        t_embed = self._get_time_embedding(timestep)
        return self.ffn(t_embed)

class TimeStepSequential(nn.Sequential):
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
        return self.conv(x)

class UNet_Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
class UNet_Encoder(nn.Module):
    def __init__(self, in_channels: int=4, num_heads: int=8, t_embed_dim: int=1280, cond_dim: int=768, ch_multiplier=[1, 2, 4, 4]):
        super().__init__()
        ch = 320
        
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        
        self.down = nn.ModuleList()
        in_ch_multiplier = [1] + ch_multiplier
        
        for i in range(len(ch_multiplier)):
            down = nn.Module()
            in_channels = ch * in_ch_multiplier[i]
            out_channels = ch * ch_multiplier[i]
            
            if i != len(ch_multiplier) - 1:
                block = nn.Sequential(
                    TimeStepSequential(UNet_ResBlock(in_channels, out_channels, t_embed_dim), UNet_TransformerEncoder(num_heads=num_heads, embedding_dim=out_channels // num_heads, cond_dim=cond_dim)),
                    TimeStepSequential(UNet_ResBlock(out_channels, out_channels, t_embed_dim), UNet_TransformerEncoder(num_heads=num_heads, embedding_dim=out_channels // num_heads, cond_dim=cond_dim)))
                downsample = UNet_Downsample(out_channels)
            else:
                block = nn.Sequential(
                TimeStepSequential(UNet_ResBlock(in_channels, out_channels, t_embed_dim)),
                TimeStepSequential(UNet_ResBlock(out_channels, out_channels, t_embed_dim)))
                
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
    def __init__(self, num_heads: int=8, t_embed_dim: int=1280, cond_dim: int=768, ch_multiplier=[1, 2, 4, 4]):
        super().__init__()
        ch = 320
        decoder_channels = ch_multiplier + [4]
        # [1280, 1280, 1280, 640, 320]
        # (1280, 1280), (1280, 1280), (1280, 640), (640, 320)
        self.up = nn.ModuleList()
        for i in reversed(range(len(ch_multiplier))):
            up = nn.Module()
            in_ch = decoder_channels[i + 1] * ch
            out_ch = decoder_channels[i] * ch
            if i > 0:
                mid_ch = decoder_channels[i-1] * ch
            else:
                mid_ch = ch
            if i == len(ch_multiplier) - 1:
                block = nn.Sequential(
                    TimeStepSequential(UNet_ResBlock(in_ch + out_ch, out_ch, t_embed_dim)),
                    TimeStepSequential(UNet_ResBlock(out_ch + out_ch, out_ch, t_embed_dim)),
                    TimeStepSequential(UNet_ResBlock(out_ch + mid_ch, out_ch, t_embed_dim)))
            else:
                block = nn.Sequential(
                    TimeStepSequential(UNet_ResBlock(in_ch + out_ch, out_ch, t_embed_dim), UNet_TransformerEncoder(num_heads=num_heads, embedding_dim=out_ch // num_heads, cond_dim=cond_dim)), 
                    TimeStepSequential(UNet_ResBlock(out_ch + out_ch, out_ch, t_embed_dim), UNet_TransformerEncoder(num_heads=num_heads, embedding_dim=out_ch // num_heads, cond_dim=cond_dim)),
                    TimeStepSequential(UNet_ResBlock(out_ch + mid_ch, out_ch, t_embed_dim), UNet_TransformerEncoder(num_heads=num_heads, embedding_dim=out_ch // num_heads, cond_dim=cond_dim)))
                
            if i != 0:
                upsample = UNet_Upsample(out_ch)
            else:
                upsample = nn.Identity()

            up.block = block
            up.upsample = upsample

            self.up.append(up)

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor], t_embed: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (b, c, h, w)
        for up in self.up:
            for layer in up.block:
                x = torch.cat([x, skip_connections.pop()], dim=1)
                x = layer(x, t_embed, cond)
            x = up.upsample(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels: int=4, out_channels: int=4, num_heads: int=8, t_embed_dim: int=320, cond_dim: int=768):
        super().__init__()
        self.time_embedding = TimeEmbedding(t_embed_dim)
        self.encoder = UNet_Encoder(in_channels=in_channels, num_heads=num_heads, t_embed_dim=t_embed_dim * 4, cond_dim=cond_dim)
        self.bottleneck = TimeStepSequential(
            UNet_ResBlock(1280, 1280, t_embed_dim * 4),
            UNet_TransformerEncoder(num_heads=8, embedding_dim=160, cond_dim=cond_dim),
            UNet_ResBlock(1280, 1280, t_embed_dim * 4)
        )
        self.decoder = UNet_Decoder(num_heads=num_heads, t_embed_dim=t_embed_dim * 4, cond_dim=cond_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(32, 320),
            nn.SiLU(),
            nn.Conv2d(320, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor, timestep: int, cond: torch.Tensor) -> torch.Tensor:

        # t: int -> (1, 1280)
        t_embed = self.time_embedding(timestep)
        x, skip_connections = self.encoder(x, t_embed, cond)
        x = self.bottleneck(x, t_embed, cond)
        x = self.decoder(x, skip_connections, t_embed, cond)
        output = self.output(x)
        return output
        