import torch
from torch import nn
from torch.nn import functional as F
class QuickGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.mul = torch.nn.quantized.FloatFunctional()
        self.mul_scalar = torch.nn.quantized.FloatFunctional()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mul.mul(x, torch.sigmoid(self.mul_scalar.mul_scalar(x, 1.702)))
        return x

class GeGELU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels * 2)

        self.mul = torch.nn.quantized.FloatFunctional()
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        x = self.mul.mul(x ,self.gelu(gate))
        return x