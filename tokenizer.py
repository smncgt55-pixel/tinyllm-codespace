from dataclasses import dataclass


@dataclass
class CharTokenizer:
    text: str | None = None
    vocab: dict[str, int] | None = None

    def __post_init__(self):
        if self.text is None and self.vocab is None:
            raise ValueError("Either text or vocab must be provided.")

        if self.vocab is None:
            chars = sorted(set(self.text))
            tokens = ["<PAD>", "<UNK>"] + chars
            self.vocab = {token: idx for idx, token in enumerate(tokens)}

        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        self.pad_id = self.vocab.get("<PAD>", 0)
        self.unk_id = self.vocab.get("<UNK>", 1)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv_vocab.get(i, "<UNK>") for i in ids)

    def __len__(self):
        return self.vocab_size
