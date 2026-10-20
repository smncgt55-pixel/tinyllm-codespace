import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model import MiniGPT
from tokenizer import CharTokenizer


def load_corpus(path: str):
    return Path(path).read_text(encoding="utf-8")


def build_sequences(corpus: str, block_size: int):
    tokenizer = CharTokenizer(corpus)
    tokens = tokenizer.encode(corpus)
    sequences = []
    for i in range(0, len(tokens) - block_size, block_size):
        seq = tokens[i : i + block_size]
        sequences.append(seq)
    if not sequences:
        raise ValueError("Corpus too short to build training windows.")
    return tokenizer, sequences


def train_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sample_text.txt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    corpus = load_corpus(args.data)
    tokenizer, sequences = build_sequences(corpus, args.block_size)
    vocab_size = tokenizer.vocab_size

    model = MiniGPT(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for seq in sequences:
            inputs = torch.tensor(seq[:-1], dtype=torch.long).unsqueeze(0)
            targets = torch.tensor(seq[1:], dtype=torch.long)

            logits = model(inputs)
            logits = logits.squeeze(0)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(sequences), 1)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:03d} | loss {avg_loss:.4f}")

    torch.save(model.state_dict(), save_dir / "mini_gpt.pt")
    print(f"Saved model to {save_dir / 'mini_gpt.pt'}")


if __name__ == "__main__":
    train_model()
