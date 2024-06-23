import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension
    bias: bool = True  # whether to use bias in attention layer.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FeedForwardBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # W_1 and b_1
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # W_2 and b_2
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        # (batch_size, seq_len, emb_size) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, emb_size)
        x = self.c_proj(self.gelu(self.c_fc(x)))

        # scaling the weights of residual stream mentioned in 2.3.Model of the GPT2 paper.
        # https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L49
        self.c_proj.NANOGPT_SCALE_INIT = 1

        return x


class CausalMultiHeadAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)  # W_q, W_k, W_v
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        # https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L21
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

    def forward(self, x):
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
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalMultiHeadAttention(config)
        self.mlp = FeedForwardBlock(config)  # also known as MLP.

    def forward(self, x):
        # mentioned in the paper Language Models are Unsupervised Multitask Learners
        # in section 2.3, layer normalization is applied before the attention & FFN.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing between token embedding and output layer.
        self.transformer.wte.weight = self.lm_head.weight

        # initialize weights.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights of the model based on:
        https://github.com/openai/gpt-2/blob/master/src/model.py
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # weights of residual stream mentioned in 2.3.Model in paper.
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None):
        # input idx is of shape (batch_size, seq_len)
        seq_len = idx.size(1)
        assert (
            seq_len <= self.config.block_size
        ), "Cannot forward, model block size is exhausted."

        # concatenate token embedding and positional embedding
        # (batch_size, seq_len, emb_size) + (seq_len, emb_size) --> (batch_size, seq_len, emb_size)
        h = self.transformer.wte(idx) + self.transformer.wpe(
            torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        )

        # apply transformer blocks as shown in ./notebooks/gpt.ipynb
        for block in self.transformer.h:
            h = block(h)

        # (batch_size, seq_len, emb_size) --> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(self.transformer.ln_f(h))
        loss = None
        if targets is not None:
            # calculate the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )  # (batch_size * seq_len, vocab_size), (batch_size * seq_len)
        return logits, loss

    # code taken from Andrej Karpathy's nanoGPT:
    # https://github.com/karpathy/nanoGPT/blob/master/model.py#L206C5-L261C21
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)

        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        print("initializing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints

        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
