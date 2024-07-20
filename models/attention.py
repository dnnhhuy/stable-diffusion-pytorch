import torch
from torch import nn
import math
import logging

class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_heads: int, embedding_dim: int, use_bias=True):
        super().__init__()
        self.proj_q = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.proj_k = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.proj_v = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)
        self.num_heads = num_heads
        self.head_dim = embedding_dim // self.num_heads
        self.proj_out = nn.Linear(embedding_dim, embedding_dim, bias=use_bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lookahead_mask: bool=True) -> torch.Tensor:
        # q, k, v: (batch_size, seq_len, embedding_dim)
        
        batch_size, seq_len, embedding_dim = q.shape
        
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        desired_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(desired_shape).permute(0, 2, 1, 3)
        k = k.view(desired_shape).permute(0, 2, 1, 3)
        v = v.view(desired_shape).permute(0, 2, 1, 3)

       

        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len) -> (batch_size, seq_len, seq_len, seq_len)
        attn_weights = q @ k.transpose(-1, -2)
        if lookahead_mask:
            mask = torch.ones_like(attn_weights, dtype=torch.bool).triu(1)
            attn_weights.masked_fill_(mask, -torch.inf)
            
        attn_weights /= math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        attn_weights = attn_weights @ v

        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, embedding_dim)
        attn_weights = attn_weights.transpose(1, 2).reshape((batch_size, seq_len, embedding_dim))

        out = self.proj_out(attn_weights)

        return out
