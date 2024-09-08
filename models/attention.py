import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
import math

class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, cond_dim: int=None, qkv_bias=True, proj_out_bias=True):
        super().__init__()
        
        if not cond_dim:
            cond_dim = embedding_dim
            
        self.proj_q = nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias)
        self.proj_k = nn.Linear(cond_dim, embedding_dim, bias=qkv_bias)
        self.proj_v = nn.Linear(cond_dim, embedding_dim, bias=qkv_bias)
        
        self.num_heads = num_heads
        self.head_dim = embedding_dim // self.num_heads
        self.proj_out = nn.Linear(embedding_dim, embedding_dim, bias=proj_out_bias)
    
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor=None, lookahead_mask: bool=False) -> torch.Tensor:
        # x: (n, seq_len, embedding_dim)
        # cond: (n, seq_len, cond_dim)
        
        batch_size, seq_len, embedding_dim = x.shape
       
        if cond is None:
            cond = x
            
        if len(cond.shape) < len(x.shape):
            cond = cond.unsqueeze(1)
        
        q = self.proj_q(x)
        k = self.proj_k(cond)
        v = self.proj_v(cond)
            
        # (batch_size, seq_len, embedding_dim) -> (n, seq_len, num_heads, head_dim) -> (n, num_heads, seq_len, head_dim)
        q = q.view(*q.shape[:2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # (n, num_heads, seq_len, head_dim) @ (n, num_heads, head_dim, seq_len) -> (n, num_heads, seq_len, seq_len)
        attn_weights = q @ k.transpose(-1, -2)

        if lookahead_mask:
            mask = torch.ones(attn_weights.shape, dtype=torch.bool).triu(1).to(x.device)
            attn_weights.masked_fill_(mask, -torch.inf)

        attn_weights = attn_weights / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # (n, num_heads, seq_len, seq_len) @ (n, num_heads, seq_len, head_dim) -> (n, num_heads, seq_len, head_dim)
        attn_weights = attn_weights @ v
        
        # (n, num_heads, seq_len, head_dim) -> (n, seq_len, num_heads, head_dim) -> (n, seq_len, embedding_dim)
        attn_weights = attn_weights.transpose(1, 2).reshape((batch_size, seq_len, embedding_dim))

        out = self.proj_out(attn_weights)
        return out