import os
import json
from dataclasses import dataclass

import torch
from torch import nn
from torch.types import *

from .attention import MultiheadSelfAttention
from safetensors.torch import load_file, load
    
@dataclass
class CLIPTextConfig:
    attention_dropout: float = 0.0
    bos_token_id: int = 0
    dropout: float = 0.0
    eos_token_id: int = 2
    hidden_act: str = "gelu"
    hidden_size: int = 1024
    initializer_factor: float = 1.0
    initializer_range: float = 0.02
    intermediate_size: int = 4096
    layer_norm_eps: float = 1e-05
    max_position_embeddings: int = 77
    num_attention_heads: int = 16
    num_hidden_layers: int = 23
    pad_token_id: int = 1
    projection_dim: int = 512
    torch_dtype: str = "float32"
    vocab_size: int = 49408
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            attention_dropout = data['attention_dropout'],
            bos_token_id = data['bos_token_id'],
            dropout = data['dropout'],
            eos_token_id = data['eos_token_id'],
            hidden_act = data['hidden_act'],
            hidden_size = data['hidden_size'],
            initializer_factor = data['initializer_factor'],
            initializer_range = data['initializer_range'],
            intermediate_size = data['intermediate_size'],
            layer_norm_eps = data['layer_norm_eps'],
            max_position_embeddings = data['max_position_embeddings'],
            num_attention_heads = data['num_attention_heads'],
            num_hidden_layers = data['num_hidden_layers'],
            pad_token_id = data['pad_token_id'],
            projection_dim = data['projection_dim'],
            torch_dtype = data['torch_dtype'],
            vocab_size = data['vocab_size']
        )
        
class TextEmbedding(nn.Module):
    def __init__(self, n_vocab: int, embed_dim: int, max_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[-1]
        position_ids = self.position_ids[:, :seq_length]
        
        input_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        embeddings = input_embeds + position_embeds
        return embeddings
        
    
class MultiLayerPercepton(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.act_layer = nn.GELU()
        self.fc2  = nn.Linear(ffn_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.fc2(x)
        return x
        
class TransformerLayer(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(normalized_shape=cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = MultiLayerPercepton(cfg.hidden_size, cfg.intermediate_size)
        self.self_attn = MultiheadSelfAttention(num_heads=cfg.num_attention_heads, embedding_dim=cfg.hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connection = x
        x = self.layer_norm1(x)
        x = self.self_attn(x=x, lookahead_mask=True)
        
        x = x + skip_connection
        skip_connection = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        
        output = x + skip_connection
        return output
        
    
class TransformerEncoder(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(cfg=cfg) for _ in range(cfg.num_hidden_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
class CLIPTextModel(nn.Module):
    def __init__(self, cfg: CLIPTextConfig=None):
        super().__init__()
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = CLIPTextConfig()
        self.embeddings = TextEmbedding(n_vocab=self.cfg.vocab_size, embed_dim=self.cfg.hidden_size, max_len=self.cfg.max_position_embeddings)
        self.encoder = TransformerEncoder(cfg=self.cfg)
        self.final_layer_norm = nn.LayerNorm(self.cfg.hidden_size, eps=self.cfg.layer_norm_eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.encoder(x)
        x = self.final_layer_norm(x)
        return x

class OpenCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = CLIPTextModel()
  
    @staticmethod
    def from_pretrained(text_encoder_pretrained_dir: str="", image_encoder_pretrained_path: str="", device: str='cpu'):
        cliptext_config_dict = json.load(open(os.path.join(text_encoder_pretrained_dir, "config.json")))
        text_encoder_cfg = CLIPTextConfig.from_dict(cliptext_config_dict)
        
        text_encoder_model_dict = load_file(os.path.join(text_encoder_pretrained_dir, "model.safetensors"), device=device)
        if "text_model.embeddings.position_ids" in text_encoder_model_dict:
            text_encoder_model_dict.pop("text_model.embeddings.position_ids")
        clip_model = OpenCLIP()
        clip_model.text_model = CLIPTextModel(cfg=text_encoder_cfg)
        clip_model.load_state_dict(text_encoder_model_dict, strict=True) 
        
        return clip_model
        
        
    def encode_image(self):
        pass
    
    def encode_text(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.text_model(input_ids)