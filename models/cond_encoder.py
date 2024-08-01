import torch
from torch import nn
from .attention import MultiheadSelfAttention
from .activation_fn import QuickGELU

class TextEncoder(nn.Module):
    def __init__(self, n_vocab: int=49408, embed_dim: int=768, max_len: int=77):
        super().__init__()
        self.text_embedding = TextEmbedding(n_vocab=n_vocab, embed_dim=embed_dim, max_len=max_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(num_heads=12, embed_dim=embed_dim, ffn_dim=embed_dim*4) for _ in range(12)
        ])
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        ## TODO: Padding text before putting into embedding
        x = x.type(torch.long)
        x = self.text_embedding(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.layernorm(x)
        return x
        
class TextEmbedding(nn.Module):
    def __init__(self, n_vocab: int, embed_dim: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, embed_dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x + self.positional_encoding
        return x
        
class TransformerEncoder(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, ffn_dim: int, dropout: float=0.0):
        super().__init__()
        self.attn_1 = MultiheadSelfAttention(num_heads=num_heads, embedding_dim=embed_dim)
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.layernorm_1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            QuickGELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.dropout_2 = nn.Dropout(dropout, inplace=True)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connection = x
        x = self.layernorm_1(x)
        x = self.attn_1(x=x.type(torch.float), lookahead_mask=True)
        x = self.dropout_1(x)
        x += skip_connection

        skip_connection = x
        x = self.layernorm_2(x)
        x = self.ffn(x)
        x = self.dropout_2(x)
        output = x + skip_connection
        return output

class ClassEncoder(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int=768):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, emb_dim)

    def forward(self, x: torch.Tensor):
        return self.embedding(x)
