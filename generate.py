import argparse
from pathlib import Path

import torch

from model import MiniGPT
from tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Artificial intelligence")
    parser.add_argument("--data", type=str, default="data/sample_text.txt")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    corpus = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(corpus)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=128,
        n_head=4,
        n_layer=4,
        block_size=64,
    )
    model.load_state_dict(torch.load("checkpoints/mini_gpt.pt", map_location="cpu"))
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    generated = model.generate(prompt_ids, max_new_tokens=args.steps, temperature=args.temperature)
    output = tokenizer.decode(generated)
    print(output)


if __name__ == "__main__":
    main()
