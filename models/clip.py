import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from .attention import MultiheadSelfAttention
from .activation_fn import QuickGELU

class TextEncoder(nn.Module):
    def __init__(self, n_vocab: int=49408, embed_dim: int=768, max_len: int=77):
        super().__init__()
        self.text_embedding = TextEmbedding(n_vocab=n_vocab, embed_dim=embed_dim, max_len=max_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(num_heads=12, embed_dim=embed_dim, ffn_dim=embed_dim*4) for _ in range(12)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm.qconfig = None
    
    def gradient_checkpointing_enabled(self, enabled=False):
        for name, module in self.encoder_layers.named_modules():
            if isinstance(module, TransformerEncoder):
                module.gradient_checkpointing = enabled

    def enable_flash_attn(self):
        for name, module in self.encoder_layers.named_modules():
            if isinstance(module, MultiheadSelfAttention):
                module.use_flash_attention = True
                
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        x = x.type(torch.long)
        x = self.text_embedding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.final_layer_norm(x)
        return x
        
class TextEmbedding(nn.Module):
    def __init__(self, n_vocab: int, embed_dim: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]

        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.embedding(input_ids.to(self.embedding.weight.device))

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings
        
class TransformerEncoder(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int, ffn_dim: int, dropout: float=0.0):
        super().__init__()
        
        self.self_attn = MultiheadSelfAttention(num_heads=num_heads, embedding_dim=embed_dim)
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.layernorm_1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            QuickGELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.dropout_2 = nn.Dropout(dropout, inplace=True)
        
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connection = x
        x = self.layernorm_1(x)
        if self.gradient_checkpointing:
            x = checkpoint.checkpoint(self.attn_1, x, lookahead_mask=True, use_reentrant=False)
        else:
            x = self.self_attn(x=x, lookahead_mask=True)
        x = self.dropout_1(x)
        
        x = x + skip_connection
        
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
