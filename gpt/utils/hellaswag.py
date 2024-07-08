"""
Download and evaluate the HellaSwag dataset.
Code taken from Andrej Karpathy's repository: https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py
"""

import os
import json
import requests
from tqdm import tqdm
import tiktoken


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


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        download_hellaswag(split)
