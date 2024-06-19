import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    emb_size: int = 768
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FeedForwardBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(self.emb_size, self.emb_size * 4)  # W_1 and b_1
        self.c_proj = nn.Linear(self.emb_size * 4, self.emb_size)  # W_2 and b_2
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        # (batch_size, seq_len, emb_size) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, emb_size)
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_size)
        self.ln_2 = nn.LayerNorm(config.emb_size)
        self.attn = MaskedMultiHeadAttention(config)
        self.mlp = FeedForwardBlock(config)  # also known as MLP.

    def forward(self, x, mask=None):
        # mentioned in the paper Language Models are Unsupervised Multitask Learners
        # in section 2.3, layer normalization is applied before the attention & FFN.
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.emb_size),
                wpe=nn.Embedding(config.block_size, config.emb_size),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.emb_size),
            )
        )
        self.lm_head = nn.Linear(config.emb_size, config.vocab_size, bias=False)
