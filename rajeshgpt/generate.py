"""
generate.py — load a trained RajeshGPT checkpoint and sample text from it.

Run:
    python generate.py --prompt "ROMEO:" --tokens 300
"""
import argparse
import torch
from model import RajeshGPT


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = RajeshGPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stoi"], ckpt["itos"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rajeshgpt.pt")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load_model(args.checkpoint, device)

    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
    text = "".join(itos[i] for i in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
