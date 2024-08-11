import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    embedding_dim: int = 768
    vocab_size: int = 50257
    layer_size: int = 12
    head_size: int = 12
    context_size: int = 64


class SelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        self().__init__()
        self.config = config

        self.qkv_embedding = nn.Embedding(
            config.embedding_dim, config.embedding_dim * 3
        )
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, x):
        pass


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        self().__init__()
        self.config = config

        self.input = nn.Linear(config.embedding_dim, config.embedding_dim * 4)
        self.activation = nn.GELU(approximate="tanh")
        self.output = nn.Linear(config.embedding_dim * 4, config.embedding_dim)

    def forward(self, x):
        pass


class AttentionLayer(nn.Module):
    def __init__(self, config: GPTConfig):
        self().__init__()

        self.self_attention_norm = nn.LayerNorm(config.embedding_dim)
        self.self_attention = SelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        pass


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        self().__init__()
        self.config = config

        self.word_encoding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_encoding = nn.Embedding(
            config.context_size, config.embedding_dim
        )
        self.attention_layers = nn.ModuleList(
            [AttentionLayer(config) for _ in range(config.layer_size)]
        )
        self.decoding_norm = nn.LayerNorm(config.embedding_dim)
        self.word_decoding = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, x):
        pass
