from typing import List, Mapping
import numpy as np
import pickle as pkl


class Tokenizer:
    def __init__(
        self, special_tokens=["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]"]
    ):
        self.vocab = {}

        # this keeps track of any equivalent tokens that are equivalent to ones
        # already in the vocab Mapping from new_token ->
        # self.vocab[existing_token]
        self.translations = {}

        # provides a mapping from token ID to all different translations of that token
        self.rev = {}

        # provides a mapping from token ID to all translations
        self.rev_translations = {}

        self.add_tokens(special_tokens)

    def encode(self, tokens: List[str]) -> List[int]:
        return np.array([self[tok] for tok in tokens])

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.rev[i] for i in token_ids]

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add tokens to the tokenizer.
        """
        for token in tokens:
            if token not in self.vocab:
                pos = len(self.vocab)
                self.vocab[token] = pos
                self.rev[pos] = token

    def add_translations(self, translations, description):
        """
        translations: dict of {new_token: existing_token}
        """
        assert all(
            key in self.vocab for key in translations.values()
        ), "Not all existing tokens are in the vocab."
        assert description != "root", "Cannot use 'root' as a description."

        self.translations.update(
            {
                new_token: self.vocab[existing_token]
                for new_token, existing_token in translations.items()
            }
        )

        for new_token, existing_token in translations.items():
            if self.vocab[existing_token] not in self.rev_translations:
                self.rev_translations[self.vocab[existing_token]] = {
                    "root": existing_token,
                    description: new_token,
                }
            else:
                assert (
                    description not in self.rev_translations[self.vocab[existing_token]]
                )
                self.rev_translations[self.vocab[existing_token]][
                    description
                ] = new_token

    def __len__(self) -> int:
        return len(self.vocab)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pkl.dump(self, f)

    def signature(self) -> int:
        # TODO maybe just sum the dictionary hashes themselves?
        return hash(
            sorted(tuple(self.vocab.items()))
            + sorted(tuple(self.translations.items()))
            + sorted(tuple(self.rev_translations.items()))
        )

    @staticmethod
    def load(path: str) -> None:
        with open(path, "rb") as inf:
            return pkl.load(inf)

    def __getitem__(self, key: str) -> int:
        if key in self.vocab:
            return self.vocab[key]
        elif key in self.translations:
            return self.translations[key]
        else:
            raise KeyError(f"Token '{key}' not found in translation or vocab.")

    def __contains__(self, key: str) -> bool:
        return key in self.vocab or key in self.translations

    # def __str__(self) -> str:
    #     pass
