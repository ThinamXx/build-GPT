"""
Download and evaluate the HellaSwag dataset.
Code taken from Andrej Karpathy's repository: https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py
"""

import os
import json
import requests
from tqdm import tqdm
import tiktoken

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel


local_dir = "/home/ubuntu/bin/build-GPT/data"  # local directory to save the data.
DATA_CACHE_DIR = os.path.join(
    local_dir, "hellaswag"
)  # create the cache local directory.

# list of URLs to download the files.
hellaswag_urls = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# initializing the tokenizer.
tokenizer = tiktoken.get_encoding("gpt2")


def download_file(url: str, fname: str, chunk_size: int = 1024):
    """Function to download a file from a URL."""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_hellaswag(split: str):
    """Function to download the Hellaswag dataset."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswag_urls[split]
    data_fname = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_fname):
        print(f"Downloading {split} split of Hellaswag dataset...")
        download_file(data_url, data_fname)


def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates).
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods).
    - label (the index of the correct completion, which we hope has the highest likelihood).
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data rendering.
    data = {"label": label, "ctx_tokens": None, "endings_tokens": []}

    # gather all the tokens.
    ctx_tokens = tokenizer.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokenizer.encode(" " + end)  # add a space because GP2 tokenizer.
        tok_row = ctx_tokens + end_tokens
        tok_rows.append(tok_row)
        mask_row = [0] * len(ctx_tokens) + [1] * len(end_tokens)
        mask_rows.append(mask_row)
        data["endings_tokens"].append(end_tokens)

    # padding the shorter sequences.
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_example(split):
    """Function to iterate over the examples in the Hellaswag dataset."""
    download_hellaswag(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


@torch.no_grad()
def evaluate_model(model_type, device="cuda"):
    """Function to evaluate the model on the Hellaswag dataset."""

    torch.set_float32_matmul_precision("high")  # set the TF32 precision.
    model = GPT2LMHeadModel.from_pretrained(model_type).to(device)
    model = model.eval()  # set the model to evaluation mode.
    # model = torch.compile(model)  # compile the model.

    num_correct = 0
    num_correct_norm = 0
    num_total = 0
    for example in iterate_example("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # enable autocast to use bfloat16 as mentioned here:
        # https://pytorch.org/docs/stable/amp.html#autocasting
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(tokens).logits  # (batch, seq_len, vocab_size)

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
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate the statistics.
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(
            f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}"
        )

        # print the example.
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_type", type=str, default="gpt2", help="model type to use."
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="device to use."
    )
    args = parser.parse_args()
    evaluate_model(args.model_type, args.device)
