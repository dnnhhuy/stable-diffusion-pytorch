import torch
from torch import nn
import math
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
    

class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, cond_dim: int=None, qkv_bias=True, proj_out_bias=True, dropout: float=0.0):
        super().__init__()
        
        if not cond_dim:
            cond_dim = embedding_dim
            
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias)
        self.key = nn.Linear(cond_dim, embedding_dim, bias=qkv_bias)
        self.value = nn.Linear(cond_dim, embedding_dim, bias=qkv_bias)
        
        self.num_heads = num_heads
        self.head_dim = embedding_dim // self.num_heads
        self.proj_attn = nn.Linear(embedding_dim, embedding_dim, bias=proj_out_bias)
        self.use_flash_attention = False
        self.dropout = dropout
    
    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lookahead_mask: bool):
        batch_size, seq_len, embedding_dim = q.shape
        # (batch_size, seq_len, embedding_dim) -> (n, seq_len, num_heads, head_dim) -> (n, num_heads, seq_len, head_dim)
        q = q.view(*q.shape[:2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # (n, num_heads, seq_len, head_dim) @ (n, num_heads, head_dim, seq_len) -> (n, num_heads, seq_len, seq_len)
        attn_weights = q @ k.transpose(-1, -2)

        if lookahead_mask:
            mask = torch.ones(attn_weights.shape, dtype=torch.bool).triu(1).to(q.device)
            attn_weights.masked_fill_(mask, -torch.inf)

        attn_weights = attn_weights / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # (n, num_heads, seq_len, seq_len) @ (n, num_heads, seq_len, head_dim) -> (n, num_heads, seq_len, head_dim)
        attn_weights = attn_weights @ v
        
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
        
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor=None, lookahead_mask: bool=False) -> torch.Tensor:
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
            
        