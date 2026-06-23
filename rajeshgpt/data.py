"""
data.py — character-level tokenizer + train/val split for RajeshGPT.

Why character-level: zero dependencies, trivially reproducible, and on
CPU-scale models the difference vs. BPE in *learnability* is small —
you still see real grammar and Shakespeare-flavored text emerge.
"""
import torch

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_data(path: str, val_fraction: float = 0.1):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    n = int(len(data) * (1 - val_fraction))
    train_data, val_data = data[:n], data[n:]
    return train_data, val_data, tok


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample a random batch of (x, y) sequences for next-token prediction."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
