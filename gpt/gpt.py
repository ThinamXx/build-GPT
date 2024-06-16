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


class TransformerBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_size)
        self.ln_2 = nn.LayerNorm(config.emb_size)
        self.attn = MaskedMultiHeadAttention(config)
        self.mlp = FeedForwardLayer(config) # also known as MLP. 

    def forward(self, x, mask=None):
        # mentioned in the paper Language Models are Unsupervised Multitask Learners
        # in section 2.3, layer normalization is applied before the attention.
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
