import os
import tiktoken
import time
import math

import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from gpt.model import GPT, GPTConfig


class DataLoader:
    """Distributed DataLoader for GPT training."""

    def __init__(self, batch_size, seq_len, process_rank, num_processes):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open("data/input.txt", "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(text)
        self.tokens = torch.tensor(tokens)
        if self.process_rank == 0:
            print(f"total tokens: {len(self.tokens)}")

        # state variables.
        self.cur_pos = self.batch_size * self.seq_len * self.process_rank

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        tokens_tensors = self.tokens[self.cur_pos : self.cur_pos + B * T + 1]
        x = (tokens_tensors[:-1]).view(B, T)  # (batch, seq_len)
        y = (tokens_tensors[1:]).view(B, T)
        self.cur_pos += B * T * self.num_processes  # move the pointer.

        if self.cur_pos + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.cur_pos = self.batch_size * self.seq_len * self.process_rank

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
    # initialize the configurations.
    max_lr = 6e-4
    warmup_steps = 10
    max_steps = 50

    # setting up the distributed data parallel (DDP).
    # torchrun sets up the environment variables for RANK, LOCAL_RANK, WORLD_SIZE.
    ddp = int(os.environ.get("RANK", -1)) != -1  # check if DDP is enabled.
    if ddp:
        # we setup the device as per the LOCAL_RANK.
        assert torch.cuda.is_available(), "DDP requires CUDA."
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do the logging, checkpointing, etc.
    else:
        # no DDP, so we assume single GPU training.
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # initialize the device.
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    # set the device type for the model because we need "cuda"  for autocast.
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # set the seed for reproducibility.
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2024)

    # initialize the gradient accumulation.
    total_batch_size = 262144  # 2^18 which is the multiple of 2.
    B = 16  # batch size.
    T = 1024  # sequence length.
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "total_batch_size must be divisible by (B * T * ddp_world_size)."
    gradient_accumulation_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(
            f"total_batch_size: {total_batch_size} => gradient_accumulation_steps: {gradient_accumulation_steps}"
        )

    # initialize the data loader.
    train_loader = DataLoader(
        batch_size=B,
        seq_len=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
    )

    torch.set_float32_matmul_precision(
        "high"
    )  # set "high" if GPU supports TF32 for 8x throughput.

    # initialize the model.
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model  # get the underlying model.

    optimizer = raw_model.configure_optimizer(
        weight_decay=0.1, lr=6e-4, device=device_type, process_rank=ddp_rank
    )
    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(gradient_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # enable autocast to use bfloat16 as mentioned here:
            # https://pytorch.org/docs/stable/amp.html#autocasting
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # scale the loss for gradient accumulation, since the loss should be MEAN of the losses
            # across the micro-steps, not the SUM of the losses.
            loss = loss / gradient_accumulation_steps
            loss_accum += loss.detach()  # detach the loss to avoid memory leak.
            if ddp:
                # while using DDP, we need to sync the gradients only at the last micro-step rather
                # than at every micro-step which is the default behavior of loss.backward().
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            loss.backward()

        if ddp:
            # all-reduce the gradients across all the processes.
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # global gradient norm clipping as mentioned in GPT-3 paper at Appendix B, page 43.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # cosine learning rate schedule as mentioned in GPT-3 paper at Appendix B, page 43.
        lr = cosine_lr_schedule(max_lr, warmup_steps, max_steps, step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()  # wait for the computation to finish.
        t1 = time.time()
        dt = (t1 - t0) * 1000  # in milliseconds.
        tokens_per_sec = (
            train_loader.batch_size
            * train_loader.seq_len
            * gradient_accumulation_steps
            * ddp_world_size
        ) / (
            t1 - t0
        )  # number of tokens processed per second.
        if master_process:
            print(
                f"step: {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )

    # destroy the process group if DDP is enabled.
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    train()

# instructions to run the code.
# simple run with one GPU: python train_gpt.py
# run with DDP: torchrun --standalone --nproc_per_node=4 train_gpt.py
