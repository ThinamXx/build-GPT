import math
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
        self.c_fc = nn.Linear(self.emb_size, 4 * self.emb_size)  # W_1 and b_1
        self.c_proj = nn.Linear(4 * self.emb_size, self.emb_size)  # W_2 and b_2
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        # (batch_size, seq_len, emb_size) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, emb_size)
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x


class CausalMultiHeadAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.emb_size % config.n_head == 0

        self.n_embd = config.emb_size
        self.n_head = config.n_head

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)  # W_q, W_k, W_v
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)

        # code taken from HF:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L142C9-L148C10
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((config.block_size, config.block_size), dtype=torch.bool)
            ).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def attention(self, query, key, value, seq_len):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) @ (batch_size, h, d_k, seq_len) --> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_scores = attention_scores.masked_fill(
            self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
        )
        attention_scores = F.softmax(attention_scores, dim=-1)
        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, x, mask=None):
        B, S, C = x.size()  # batch_size, seq_len, emb_size
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)

        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k) --> (batch_size, h, seq_len, d_k)
        query = query.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
        key = key.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
        value = value.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)

        x, _ = self.attention(query, key, value, S)

        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(B, S, C)

        x = self.c_proj(x)  # (batch_size, seq_len, d_model)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_size)
        self.ln_2 = nn.LayerNorm(config.emb_size)
        self.attn = CausalMultiHeadAttention(config)
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
