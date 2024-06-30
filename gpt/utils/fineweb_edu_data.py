"""Fine Web Edu Data: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
This script downloads, tokenizes, and saves the data shards to disk.
"""

import os
import tiktoken
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset

local_dir = "/home/ubuntu/bin/build-GPT/data"  # local directory to save the data.
dataset_name = "sample-10BT"  # dataset name.
shard_size = int(1e8)  # 100M in each shard, total of 100 shards. 10B / 100 = 100M.

# create the cache local directory.
DATA_CACHE_DIR = os.path.join(local_dir, "edu_fineweb10B")
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset.
fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name=dataset_name, split="train")

# tokenize the dataset.
tokenizer = tiktoken.get_encoding("gpt2")
eot = tokenizer._special_tokens["<|endoftext|>"]  # end of text token for GPT-2.


def tokenize(doc):
    # tokenize the document and returns a numpy array of uint16 tokens.
    tokens = [eot]  # the special tok "<|endoftext|>" delimits the documents.
    tokens.extend(tokenizer.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "tokens too large for uint16."
    tokens_np_16 = tokens_np.astype(np.uint16)
    return tokens_np_16


# function to save the data shard.
def save_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# tokenize all the documents, and save output shards, each shard is 100M.
nprocs = max(1, os.cpu_count() // 2)  # number of processes.
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty(
        (shard_size,), dtype=np.uint16
    )  # buffer to store the tokens.
    tokens_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fineweb, chunksize=16):
        if tokens_count + len(tokens) < shard_size:
            all_tokens_np[tokens_count : tokens_count + len(tokens)] = tokens
            tokens_count += len(tokens)

            # update the progress bar.
            if progress_bar is None:
                progress_bar = tqdm(
                    total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                )
            progress_bar.update(len(tokens))

        else:
            # save the current shard and reset the buffer.
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            # split the document into multiple shards.
            remaining = shard_size - tokens_count
            progress_bar.update(remaining)
            all_tokens_np[tokens_count : tokens_count + remaining] = tokens[:remaining]
            save_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # popluate the next shard with the remaining tokens.
            all_tokens_np[: len(tokens) - remaining] = tokens[remaining:]
            tokens_count = len(tokens) - remaining

    # save the last shard.
    if tokens_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        save_datafile(filename, all_tokens_np[:tokens_count])


# instructions to run the script.
# python ./gpt/utils/fineweb_edu_data.py