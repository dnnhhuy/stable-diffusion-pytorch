import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float=0.0, eps: float=1e-6):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = nn.Identity()

        self.silu_1 = nn.SiLU()
        self.silu_2 = nn.SiLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        out = self.silu_1(x_norm)
        
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.silu_2(out)
        
        out = self.dropout(out)
        out = self.conv2(out)

        x_in = self.conv_shortcut(x)
        out = out + x_in
        return out
        
