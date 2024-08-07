import tiktoken
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


def generate_text(model, prompt, max_len=30, num_return_sequences=1):
    """Function to generate text using the GPT model."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    model.eval()
    model = model.to(device)

    tokenizer = tiktoken.get_encoding("gpt2")  # get the tokenizer
    tokens = tokenizer.encode(prompt)  # encode the text

    # convert the tokens to tensor.
    tokens = torch.tensor(tokens, dtype=torch.long)  # (seq_len,)
    tokens = tokens.unsqueeze(0).repeat(
        num_return_sequences, 1
    )  # repeat the tokens for num_samples times.
    x = tokens.to(device)  # move the tokens to cuda. (batch, seq_len)

    # generate the text.
    while x.size(1) < max_len:
        with torch.no_grad():
            # enable autocast to use bfloat16 as mentioned here:
            # https://pytorch.org/docs/stable/amp.html#autocasting
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x)  # (batch, seq_len, vocab_size)
            # get the last token logits.
            logits = logits[:, -1, :]  # (batch, vocab_size)
            probs = F.softmax(logits, dim=-1)  # get the probabilities.

            # perform top-k sampling as mentioned in HF:
            # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
            top_k = 50
            topk_probs, topk_indices = torch.topk(
                probs, top_k, dim=-1
            )  # get the top-k probs and indices.
            # sample the next token from the top-k probs.
            next_token = torch.multinomial(topk_probs, num_samples=1)  # (batch, 1)
            # gather the corresponding token index.
            next_token_index = torch.gather(topk_indices, -1, next_token)  # (batch, 1)
            x = torch.cat((x, next_token_index), dim=1)

    # decode the tokens.
    for i in range(num_return_sequences):
        tokens = x[i, :max_len].tolist()
        decoded_text = tokenizer.decode(tokens)
        print(f"generated sample ==>{i+1}: {decoded_text}")


if __name__ == "__main__":
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)

    # load the model.
    model = GPT.from_pretrained("gpt2")  # loading the gpt2 checkpoints from HF.
    # model = GPT(GPTConfig())  # initialize the model with random weights.

    prompt = "Hello, I'm a language model,"
    generate_text(model, prompt, max_len=30, num_return_sequences=3)
