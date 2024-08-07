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

        self.silu_1 = nn.SiLU()
        self.silu_2 = nn.SiLU()

        self.add = torch.nn.quantized.FloatFunctional()

        self.dequant = torch.ao.quantization.DeQuantStub()
        self.quant_1 = torch.ao.quantization.QuantStub()
        self.quant_2 = torch.ao.quantization.QuantStub()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.groupnorm_1(x)
        x_norm = self.dequant(x_norm)
        out = self.silu_1(x_norm)
        out = self.quant_1(out)
        
        out = self.conv_1(out)

        out = self.groupnorm_2(out)
        out = self.dequant(out)
        out = self.silu_2(out)
        out = self.quant_2(out)
        
        out = self.dropout(out)
        out = self.conv_2(out)

        x = self.proj_input(x)
        out = self.add.add(out, x)
        return out
        
