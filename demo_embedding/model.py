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
        super().__init__()
        self.config = config
        assert (
            config.embedding_dim % config.head_size
        ) == 0, "Embedding dim must be dividble by head size."

        self.qkv_embedding = nn.Linear(config.embedding_dim, config.embedding_dim * 3)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, x):
        B, T, C = x.size()
        embedding_dim = self.config.embedding_dim
        head_size = self.config.head_size

        qkv = self.qkv_embedding(x)
        q, k, v = qkv.split(embedding_dim, dim=2)
        q = q.view(B, T, head_size, C // head_size).transpose(1, 2)
        k = k.view(B, T, head_size, C // head_size).transpose(1, 2)
        v = v.view(B, T, head_size, C // head_size).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.output(out)
        return out


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.input = nn.Linear(config.embedding_dim, config.embedding_dim * 4)
        self.activation = nn.GELU(approximate="tanh")
        self.output = nn.Linear(config.embedding_dim * 4, config.embedding_dim)

    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)
        x = self.output(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.self_attention_norm = nn.LayerNorm(config.embedding_dim)
        self.self_attention = SelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.self_attention(self.self_attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_to_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_encoding = nn.Embedding(
            config.context_size, config.embedding_dim
        )
        self.attention_layers = nn.ModuleList(
            [AttentionLayer(config) for _ in range(config.layer_size)]
        )
        self.decoding_norm = nn.LayerNorm(config.embedding_dim)
        self.embedding_to_token = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, inputs, targets=None):
        B, T = inputs.size()

        assert T <= self.config.context_size, "Context size is too large"

        pos = torch.arange(0, T, step=1, dtype=torch.long, device=inputs.device)
        pos_emb = self.positional_encoding(pos)
        tok_emb = self.token_to_embedding(inputs)
        x = tok_emb + pos_emb

        for layer in self.attention_layers:
            x = layer(x)

        x = self.decoding_norm(x)
        logits = self.embedding_to_token(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss
