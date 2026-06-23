"""
train.py — trains RajeshGPT on Tiny Shakespeare, CPU-friendly settings.

Run:
    python train.py
"""
import time
import torch
from data import load_data, get_batch
from model import RajeshGPT

# ---- config (tuned for CPU laptops — small enough to train in minutes) ----
BLOCK_SIZE = 128       # context length
BATCH_SIZE = 32
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
LR = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337

torch.manual_seed(SEED)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    print(f"device: {DEVICE}")
    train_data, val_data, tok = load_data("input.txt")
    print(f"vocab_size={tok.vocab_size}  train_tokens={len(train_data)}  val_tokens={len(val_data)}")

    model = RajeshGPT(
        vocab_size=tok.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"model params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS)

    t0 = time.time()
    for it in range(1, MAX_ITERS + 1):
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if it % EVAL_INTERVAL == 0 or it == 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            print(
                f"iter {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed"
            )

    # save model + tokenizer vocab together so generate.py / the web app
    # can rebuild everything from one file
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": tok.vocab_size,
                "block_size": BLOCK_SIZE,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
                "dropout": DROPOUT,
            },
            "stoi": tok.stoi,
            "itos": tok.itos,
        },
        "rajeshgpt.pt",
    )
    print("saved checkpoint to rajeshgpt.pt")


if __name__ == "__main__":
    main()
