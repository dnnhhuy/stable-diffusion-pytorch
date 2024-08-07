import torch
from torch import nn
from torch.nn import functional as F
from .resnet import ResidualBlock
from .attention import MultiheadSelfAttention
from typing import List

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
    def __init__(self, in_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.attn = MultiheadSelfAttention(num_heads=1, embedding_dim=in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, c, h, w)
        batch_size, channels, h, w = x.shape
        x_norm = self.groupnorm(x)
        # (n, c, h, w) -> (n, c, h * w) -> (n, h * w, c)
        x_norm = x_norm.view((batch_size, channels, -1)).transpose(1, 2)
        
        # (n, h * w, c)
        out = self.attn(x=x_norm)

        # (n, h * w, c) -> (n, c, h * w) -> (n, c, h, w)
        out = out.transpose(1, 2).view(x.shape)
        return out + x
        
class VAE_Encoder(nn.Module):
    def __init__(self, in_channels: int, ch_mult: List[int]=[1, 2, 4, 4], dropout: float=0.0, z_channels: int=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)

        # start
        self.down = nn.ModuleList()
        in_ch_mult = [1] + ch_mult
        ch = 128
        for i in range(len(ch_mult)):
            block_in = ch * in_ch_mult[i]
            block_out = ch * ch_mult[i]
            block = nn.Sequential( 
                ResidualBlock(block_in, block_out, dropout),
                ResidualBlock(block_out, block_out, dropout),
            )
            
            down = nn.Module()
            down.block = block
            if i != len(ch_mult) - 1:
                down.downsample = Downsample(block_out)
            else:
                down.downsample = nn.Identity()
                
            self.down.append(down)
            curr_channels = block_out

        # middle
        self.mid = nn.Module()
        self.mid.res_block_1 = ResidualBlock(curr_channels, curr_channels)
        self.mid.attn_block_1 = AttentionBlock(in_channels=curr_channels)
        self.mid.res_block_2 = ResidualBlock(curr_channels, curr_channels)
        
        # end
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=curr_channels),
            nn.SiLU(),
            nn.Conv2d(curr_channels, 2*z_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(2*z_channels, 2*z_channels, kernel_size=1, stride=1, padding=0))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        
        for down in self.down:
            x = down.block(x)
            x = down.downsample(x)

        x = self.mid.res_block_1(x)
        x = self.mid.attn_block_1(x)
        x = self.mid.res_block_2(x)

        x = self.out(x)
        return x


class VAE_Decoder(nn.Module):
    def __init__(self, ch_mult: List[int]=[1, 2, 4, 4], dropout: float=0.0, z_channels: int=4):
        super().__init__()

        ch = 128
        block_in = ch*ch_mult[-1]
        self.conv_in = nn.Sequential(nn.Conv2d(z_channels, z_channels, kernel_size=1, padding=0), 
                                     nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1))
        
    
        # mid
        self.mid = nn.Module()
        self.mid.res_block_1 = ResidualBlock(block_in, block_in)
        self.mid.attn_block_1 = AttentionBlock(in_channels=block_in)
        self.mid.res_block_2 = ResidualBlock(block_in, block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[i]
            block = nn.Sequential(
                ResidualBlock(block_in, block_out),
                ResidualBlock(block_out, block_out),
                ResidualBlock(block_out, block_out)
            )
            up = nn.Module()
            up.block = block
            if i != 0:
                up.upsample = UpSample(in_channels=block_out)
            else:
                up.upsample = nn.Identity()
            self.up.append(up)
            block_in = block_out

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=ch), 
            nn.SiLU(),
            nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1))

        
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, h, w)
        
        x = self.conv_in(x)

        x = self.mid.res_block_1(x)
        x = self.mid.attn_block_1(x)
        x = self.mid.res_block_2(x)

        for up in self.up:
            x = up.block(x)
            x = up.upsample(x)
        x = self.out(x)
        return x
        
        
class VAE(nn.Module):
    def __init__(self, in_channels: int=3, z_channels: int=4):
        super().__init__()
        self.encoder = VAE_Encoder(in_channels=in_channels)
        self.decoder = VAE_Decoder(z_channels=z_channels)

    def encode(self, x: torch.Tensor, noise=None) -> torch.Tensor:
        # z: (n, c, h, w)
        z = self.encoder(x)
        
        mean, log_variance = z.chunk(2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = torch.sqrt(variance)
        
        if noise is not None:
            output = mean + stdev * noise
        else:
            output = mean + stdev * torch.randn_like(stdev)
        output = output * 0.18215
        return output, mean, stdev
        

    def decode(self, z: torch.Tensor):
        z = z / 0.18215
        out = self.decoder(z) 
        return out