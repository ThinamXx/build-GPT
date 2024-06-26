import tiktoken
import time
import math

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


def cosine_lr_schedule(max_lr, warmup_steps, max_steps, cur_step):
    """Cosine learning rate schedule with warm-up as mentioned in GPT-3 paper at Appendix B, page 43.
    Code taken from: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L354C5-L364C46
    """
    min_lr = max_lr * 0.1  # 10% of max_lr.

    # 1) linear warmup for warmup_iters steps
    if cur_step < warmup_steps:
        return max_lr * (cur_step + 1) / warmup_steps

    # 2) if cur_step > lr_decay_iters, return min learning rate
    if cur_step > max_steps:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (cur_step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0

    return min_lr + coeff * (max_lr - min_lr)


def train():
    # initialize the hyperparameters.
    max_lr = 6e-4
    warmup_steps = 10
    max_steps = 50

    torch.set_float32_matmul_precision(
        "high"
    )  # set "high" if GPU supports TF32 for 8x throughput.

    # initialize the model.
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    # model = torch.compile(model)

    train_loader = DataLoader(batch_size=8, seq_len=1024)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8
    )
    for step in range(max_steps):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # enable autocast to use bfloat16 as mentioned here:
        # https://pytorch.org/docs/stable/amp.html#autocasting
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        # global gradient norm clipping as mentioned in GPT-3 paper at Appendix B, page 43.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # cosine learning rate schedule as mentioned in GPT-3 paper at Appendix B, page 43.
        lr = cosine_lr_schedule(max_lr, warmup_steps, max_steps, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()  # wait for the computation to finish.
        t1 = time.time()
        dt = (t1 - t0) * 1000  # in milliseconds.
        tokens_per_sec = (train_loader.batch_size * train_loader.seq_len) / (
            t1 - t0
        )  # number of tokens processed per second.
        print(
            f"step: {step:4d} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )


if __name__ == "__main__":
    train()
