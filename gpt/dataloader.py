import os
import numpy as np

import tiktoken
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def load_tokens(filename):
    """Function to load the tokens from the shard file."""
    tokens = np.load(filename)
    tokens = tokens.astype(np.int32)
    tok_tensor = torch.tensor(tokens, dtype=torch.long)
    return tok_tensor


class DataLoaderShakespeare:
    """DataLoader for Shakespeare dataset."""

    def __init__(self, batch_size, seq_len, process_rank, num_processes):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes

        # load the dataset.
        data_root = "/home/ubuntu/bin/build-GPT/data/input.txt"
        with open(data_root, "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"total tokens: {len(self.tokens)}")

        self.cur_pos = self.batch_size * self.seq_len * self.process_rank

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        tokens_tensors = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = (tokens_tensors[:-1]).view(B, T)  # (batch, seq_len)
        y = (tokens_tensors[1:]).view(B, T)
        self.cur_pos += B * T * self.num_processes  # move the pointer.

        if self.cur_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.cur_pos = self.batch_size * self.seq_len * self.process_rank

        return x, y


class DataLoader:
    """Distributed DataLoader for GPT training."""

    def __init__(self, batch_size, seq_len, process_rank, num_processes, split):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ["train", "val"], "split must be either 'train' or 'val'."

        # load the tokens from the shard files.
        data_root = "/root/bin/build-GPT/data/data/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split: {split}"
        if self.process_rank == 0:
            print(f"loading {len(shards)} shards for split: {split}")
        self.reset()

    def reset(self):
        self.cur_shard = 0
        self.tokens = load_tokens(self.shards[self.cur_shard])
        self.cur_pos = self.batch_size * self.seq_len * self.process_rank

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        tokens_tensors = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = (tokens_tensors[:-1]).view(B, T)  # (batch, seq_len)
        y = (tokens_tensors[1:]).view(B, T)
        self.cur_pos += B * T * self.num_processes  # move the pointer.
        # if loading the next batch crosses the shard boundary, load the next shard.
        if self.cur_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.cur_shard = (self.cur_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.cur_shard])
            self.cur_pos = self.batch_size * self.seq_len * self.process_rank

        return x, y
