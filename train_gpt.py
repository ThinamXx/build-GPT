import tiktoken
import torch
import torch.nn.functional as F

from gpt.model import GPT, GPTConfig

torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

# initialize the device.
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# read the text data.
PATH = "data/input.txt"
with open(PATH, "r") as f:
    text = f.read()
data = text[:1000]  # read the first 1000 characters.

tokenizer = tiktoken.get_encoding("gpt2")  # get the tokenizer.
tokens = tokenizer.encode(data)  # encode the text.

B, T = 3, 30  # batch size and sequence length.
tokens_tensors = torch.tensor(
    tokens[: B * T + 1], dtype=torch.long
)  # convert the tokens to tensor.
x = tokens_tensors[:-1].view(B, T)  # (batch, seq_len)
y = tokens_tensors[1:].view(B, T)  # (batch, seq_len)
x, y = x.to(device), y.to(device)

# initialize the model and get the logits.
model = GPT(GPTConfig())
model.to(device)
logits, loss = model(x, y)  # (batch, seq_len, vocab_size)
print(logits.shape)
print(loss)
