import os
import time
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import GPT, GPTConfig
from inference import generate_text
from utils.hellaswag import iterate_example, render_example


def load_tokens(filename):
    """Function to load the tokens from the shard file."""
    tokens = np.load(filename)
    tokens = tokens.astype(np.int32)
    tok_tensor = torch.tensor(tokens, dtype=torch.long)
    return tok_tensor


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


def get_most_likely_sequence(tokens, mask, logits):
    """Function to get the most likely sequence from the logits."""

    # evaluate autoregressive loss likelihoods.
    shifted_logits = (logits[..., :-1, :]).contiguous()  # skip the last token.
    shifted_tokens = (tokens[..., 1:]).contiguous()  # skip the first token.
    flat_shifted_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flat_shifted_tokens = shifted_tokens.view(-1)
    shift_loss = F.cross_entropy(
        flat_shifted_logits, flat_shifted_tokens, reduction="none"
    )
    shift_loss = shift_loss.view(tokens.size(0), -1)

    # get the average loss for each example.
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_loss = shift_loss * shift_mask
    sum_loss = masked_shift_loss.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # now get the likelihood of the correct completion.
    pred_norm = avg_loss.argmin().item()

    return pred_norm


def train():
    # initialize the configurations.
    max_lr = 6e-4
    warmup_steps = 713  # 187e6 / 262144 = 713, refer to GPT-3 paper at Appendix B.
    max_steps = 38146  # 10e9 / 262144 = 38146

    # initialize the gradient accumulation.
    total_batch_size = 262144  # 2^18 tokens per step, which is half of the GPT-3.
    B = 8  # batch size.
    T = 1024  # sequence length.

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
        split="train",
    )
    val_loader = DataLoader(
        batch_size=B,
        seq_len=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
    )

    torch.set_float32_matmul_precision(
        "high"
    )  # set "high" if GPU supports TF32 for 8x throughput.

    # initialize the model.
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)

    # compile the model for faster performance.
    use_compile = True
    if use_compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model  # get the underlying model.

    optimizer = raw_model.configure_optimizer(
        weight_decay=0.1, lr=6e-4, device=device_type, process_rank=ddp_rank
    )

    # saving the checkpoints and logs.
    log_dir = "/root/bin/build-GPT/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:
        pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        # enable the evaluation mode.
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()  # reset the val_loader.
            with torch.no_grad():
                val_loss_accum = 0.0
                val_accum_steps = 20
                for _ in range(val_accum_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    # enable autocast to use bfloat16 as mentioned here:
                    # https://pytorch.org/docs/stable/amp.html#autocasting
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_accum_steps
                    val_loss_accum += loss.detach()
            if ddp:
                # calculate the average loss across all the processes or ranks.
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(
                        f"step: {step} | validation loss: {val_loss_accum.item():.4f}\n"
                    )

                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "config": raw_model.config,
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(checkpoint, checkpoint_path)

        # evaluation of the model on the Hellaswag dataset.
        if (step % 250 == 0 or last_step) and (not use_compile):
            num_total = 0
            num_correct_norm = 0
            for i, example in enumerate(iterate_example("val")):
                # only process where i % ddp_world_size == ddp_rank.
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)  # render the example.
                tokens = tokens.to(device)
                mask = mask.to(device)

                # get the logits from the model.
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    pred_norm = get_most_likely_sequence(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)

            # accumulate the statistics.
            if ddp:
                # calculate the average loss across all the processes or ranks.
                num_total_tensor = torch.tensor(
                    num_total, dtype=torch.long, device=device
                )
                num_correct_norm_tensor = torch.tensor(
                    num_correct_norm, dtype=torch.long, device=device
                )
                dist.all_reduce(num_total_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm_tensor, op=dist.ReduceOp.SUM)
                num_total = num_total_tensor.item()
                num_correct_norm = num_correct_norm_tensor.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"hellaswag accuracy: {acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"step: {step} | hellaswag accuracy: {acc_norm:.4f}\n")

        # generate samples using the model.
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            # when using torch.compile() to generate the samples, we get errors, so
            # we skip the generation step.
            generate_text(
                model,
                "Hello, I'm a language model,",
                max_len=30,
                num_return_sequences=2,
            )

        # enable the training mode.
        model.train()
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
            # calculate the average loss across all the processes or ranks.
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
                f"step: {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"step: {step} | train loss: {loss_accum.item():.6f}\n")

    # destroy the process group if DDP is enabled.
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    train()

# instructions to run the code.
# simple run with one GPU: python ./gpt/train_gpt.py
# run with DDP: torchrun --standalone --nproc_per_node=4 ./gpt/train_gpt.py
