# TinyLlama-style mini LLM for GitHub Codespaces

This project builds a small GPT-like language model that is trainable in a GitHub Codespaces environment. It is intentionally compact, readable, and educational.

The goal is not to match frontier LLMs such as GPT-4 or Claude. Instead, the goal is to create a working mini transformer with:
- tokenization
- embeddings
- causal attention
- transformer blocks
- next-token prediction
- text generation
- code training workflow

## Model architecture

This project uses a decoder-only transformer inspired by GPT:
- token embeddings
- positional embeddings
- multi-head causal self-attention
- feed-forward network
- final language-model head

## Files

- `model.py` — transformer model
- `tokenizer.py` — simple character tokenizer
- `train.py` — trainer script
- `generate.py` — text generator
- `data/sample_text.txt` — example training corpus

## Quick start

```bash
cd /workspaces/tinyllm-codespace
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python generate.py --prompt "Artificial intelligence"
```

## What this model learns

It learns to predict the next character/token from a small text corpus. It is not a production LLM, but it is a real model that can be trained and used to generate text.

## Limits

This is a small educational model designed for Codespaces. It is not meant to replace large-scale LLM infrastructure. It is intended to teach the mechanics of modern language models in a lightweight and manageable setup.

## Example generated output

After training, the generation script can produce outputs like:

```text
Artificial intelligence learns patterns from data.
A language model predicts the next token.
Transformers use attention to focus on context.
```

## Training notes

For best results:
- use a richer corpus
- add more epochs
- increase model width
- add a larger text dataset
- use more GPU memory if available

## Credits

This project is intentionally compact and educational, designed to be understandable and extensible in a GitHub Codespaces environment.
