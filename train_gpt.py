import tiktoken
import time

import torch
import torch.nn.functional as F

from gpt.model import GPT, GPTConfig


torch.manual_seed(2024)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2024)

# initialize the device.
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")


class DataLoader:

    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len

        with open("data/input.txt", "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"total tokens: {len(self.tokens)}")
        print(f"1 epoch has {len(self.tokens) // (batch_size * seq_len)} batches.")

        # state variables.
        self.cur_pos = 0

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        tokens_tensors = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = (tokens_tensors[:-1]).view(B, T)  # (batch, seq_len)
        y = (tokens_tensors[1:]).view(B, T)
        self.cur_pos += B * T  # move the pointer.

        if self.cur_pos + (B * T + 1) >= len(self.tokens):
            self.cur_pos = 0

        return x, y


def train():
    # initialize the model.
    model = GPT(GPTConfig())
    model.to(device)

    train_loader = DataLoader(batch_size=16, seq_len=1024)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()  # wait for the computation to finish.
        t1 = time.time()
        dt = (t1 - t0) * 1000  # in milliseconds.
        print(f"step: {i}, loss: {loss.item()} time: {dt:.2f}ms")


if __name__ == "__main__":
    train()
