# RajeshGPT

A GPT-style language model built entirely from scratch — no Hugging Face,
no pretrained weights — trained on Shakespeare, with a small website that
watches it write live, character by character.

## What's in here

| File | What it does |
|---|---|
| `data.py` | Character-level tokenizer + train/val split |
| `model.py` | The transformer itself: attention, MLP, blocks, the full `RajeshGPT` class |
| `train.py` | Trains the model on `input.txt` (Tiny Shakespeare), saves `rajeshgpt.pt` |
| `generate.py` | Command-line text generation from a trained checkpoint |
| `serve.py` | Tiny local API (Flask) that streams generated text to the website |
| `index.html` | The RajeshGPT website — open this in a browser |
| `input.txt` | Training data: the complete works of Shakespeare (~1MB) |

## Quickstart

```bash
# 1. install dependencies
pip install torch flask

# 2. train the model (a few minutes on a normal laptop with the default
#    settings in train.py — 3000 iterations, ~1M-param model)
python train.py

# 3. start the local inference server (keep this running)
python serve.py

# 4. open index.html in your browser
```

That's it — type a prompt like `ROMEO:` and watch it generate.

## How it actually works

This is a **decoder-only transformer**, the same family as GPT-2/3 and
Llama, just much smaller:

- **Tokenizer**: character-level (65 unique characters in Shakespeare's
  English — way simpler than real BPE tokenizers, but the model training
  mechanics are identical)
- **Embeddings**: each character gets a learned vector, plus a learned
  vector for its position in the sequence
- **4 transformer blocks**, each with:
  - causal self-attention (a character can only "look at" characters
    before it — this is what makes it a *language* model, not just a
    pattern matcher)
  - a small MLP
  - residual connections + LayerNorm around each
- **Output head**: projects back to a probability distribution over all
  65 characters — "what comes next?"
- **Training**: cross-entropy loss between predicted and actual next
  character, AdamW optimizer, cosine learning-rate schedule

Default config is **~826K parameters** — tiny by modern standards (GPT-3
is 175 *billion*), but the architecture is the real thing at toy scale.

## Tuning it

All the knobs are at the top of `train.py`:

- `N_LAYER`, `N_HEAD`, `N_EMBD` — model size. Bigger = slower to train,
  potentially better text, more RAM.
- `BLOCK_SIZE` — how much context the model can see at once.
- `MAX_ITERS` — how long to train. Loss should keep dropping past 3000
  if you have the patience; diminishing returns set in eventually for a
  model this size.
- `BATCH_SIZE` — lower this if training is slow or you're low on RAM.

If you have a GPU, `train.py` will automatically use it (`DEVICE =
"cuda" if torch.cuda.is_available() else "cpu"`) and training will be
much faster.

## Using your own text instead of Shakespeare

Replace `input.txt` with any plain-text file (a novel, your own writing,
song lyrics you have rights to, etc.) and re-run `train.py`. The smaller
and more repetitive the text, the faster it'll pick up the style; the
larger and more varied, the longer it'll take but the more it can learn.

## Why the website streams character-by-character

That's not just a visual effect — it's literally how the model works.
Each character is sampled one at a time from the probability
distribution the model just computed, then fed back in to predict the
*next* character. The "alternatives" panel on the site shows you that
process directly: every character it writes was chosen over other
candidates it seriously considered.
