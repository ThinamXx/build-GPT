import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 6
    n_heads: int = 32
    vocab_size: int = -1  # given by tokenizer
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"