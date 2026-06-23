"""
serve.py — tiny local API server for RajeshGPT.

Loads rajeshgpt.pt and exposes a single endpoint the website calls to
generate text. Streams tokens one at a time (Server-Sent Events) so the
page can show the model "thinking" character by character, the way the
model itself actually works internally.

Run:
    python serve.py
Then open index.html (it calls http://localhost:5000/generate).
"""
import json
import torch
from flask import Flask, request, Response
from model import RajeshGPT

app = Flask(__name__)

DEVICE = "cpu"
CHECKPOINT = "rajeshgpt.pt"

_model = None
_stoi = None
_itos = None


def get_model():
    global _model, _stoi, _itos
    if _model is None:
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        cfg = ckpt["config"]
        m = RajeshGPT(
            vocab_size=cfg["vocab_size"],
            block_size=cfg["block_size"],
            n_embd=cfg["n_embd"],
            n_head=cfg["n_head"],
            n_layer=cfg["n_layer"],
            dropout=cfg["dropout"],
        )
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        _model = m
        _stoi, _itos = ckpt["stoi"], ckpt["itos"]
    return _model, _stoi, _itos


def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return add_cors(Response(status=204))

    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "\n") or "\n"
    n_tokens = max(1, min(int(data.get("tokens", 200)), 1000))
    temperature = max(0.05, min(float(data.get("temperature", 0.8)), 2.0))
    top_k = data.get("top_k", 40)
    top_k = int(top_k) if top_k else None

    model, stoi, itos = get_model()

    # unknown characters are dropped rather than erroring, so the demo
    # never breaks just because someone typed an emoji
    known_prompt = "".join(c for c in prompt if c in stoi) or "\n"
    idx = torch.tensor([[stoi[c] for c in known_prompt]], dtype=torch.long)

    def stream():
        nonlocal idx
        yield from ()  # generator marker
        with torch.no_grad():
            for _ in range(n_tokens):
                idx_cond = idx[:, -model.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
                ch = itos[next_id.item()]
                yield f"data: {json.dumps({'token': ch})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    resp = Response(stream(), mimetype="text/event-stream")
    return add_cors(resp)


@app.route("/health", methods=["GET"])
def health():
    try:
        model, _, _ = get_model()
        return add_cors(Response(
            json.dumps({"status": "ok", "params": model.num_params()}),
            mimetype="application/json",
        ))
    except FileNotFoundError:
        return add_cors(Response(
            json.dumps({"status": "no_checkpoint"}),
            mimetype="application/json",
            status=503,
        ))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
