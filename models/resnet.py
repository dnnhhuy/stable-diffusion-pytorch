import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float=0.0):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.proj_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_input = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.groupnorm_1(x)
        out = F.silu(x_norm)
        out = self.conv_1(out)

        out = self.groupnorm_2(out)
        out = F.silu(out)
        out = self.dropout(out)
        out = self.conv_2(out)
        return out + self.proj_input(x)
        
