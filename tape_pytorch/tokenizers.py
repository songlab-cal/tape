from typing import Union, List, Optional, Type
from pathlib import Path
from abc import ABC, abstractmethod, abstractproperty
import logging

import sentencepiece as spm
from tape_pytorch.registry import registry

logger = logging.getLogger(__name__)


class TAPETokenizer(ABC):
    r"""
    Abstract Tokenizer Class
    """

    @abstractproperty
    def vocab_size(self) -> int:
        return NotImplemented

    @abstractproperty
    def start_token(self) -> str:
        pass

    @abstractproperty
    def stop_token(self) -> str:
        pass

    @abstractproperty
    def mask_token(self) -> str:
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        return NotImplemented

    @abstractmethod
    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        return NotImplemented

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    @abstractmethod
    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return NotImplemented

    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.convert_id_to_token(id_) for id_ in indices]

    @abstractmethod
    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return NotImplemented

    @abstractmethod
    def add_special_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        return NotImplemented

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls()


@registry.register_tokenizer('amino_acid')
class AminoAcidTokenizer(TAPETokenizer):
    r"""
    Constructs a DummyTokenizer
    """

    TOKENS = ["<pad>", "<mask>", "<cls>", "<sep>", "<unk>",
              "A", "B", "C", "D", "E", "F", "G", "H", "I",
              "K", "L", "M", "N", "O", "P", "Q", "R", "S",
              "T", "U", "V", "W", "X", "Y", "Z"]
    VOCAB = {token: i for i, token in enumerate(TOKENS)}

    @property
    def start_token(self) -> str:
        return "<cls>"

    @property
    def stop_token(self) -> str:
        return "<sep>"

    @property
    def mask_token(self) -> str:
        return "<mask>"

    @property
    def vocab_size(self) -> int:
        return len(self.TOKENS)

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.VOCAB[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.TOKENS[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return ''.join(tokens)

    def add_special_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        cls_token = [self.convert_token_to_id(self.start_token)]
        sep_token = [self.convert_token_to_id(self.stop_token)]
        return cls_token + token_ids + sep_token

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls()


@registry.register_tokenizer('unirep')
class UniRepTokenizer(TAPETokenizer):
    r""" Constructs a UniRepTokenizer. This matches the tokenizer from
         Alley et al. https://www.biorxiv.org/content/10.1101/589333v1.
    """

    VOCAB = {
        "<pad>": 0, "M": 1, "R": 2, "H": 3, "K": 4, "D": 5, "E": 6, "S": 7, "T": 8, "N": 9,
        "Q": 10, "C": 11, "U": 12, "G": 13, "P": 14, "A": 15, "V": 16, "I": 17, "F": 18,
        "Y": 19, "W": 20, "L": 21, "O": 22, "X": 23, "Z": 23, "B": 23, "J": 23, "start": 24,
        "stop": 25}

    TOKENS = list(VOCAB.keys())

    @property
    def start_token(self) -> str:
        return "start"

    @property
    def stop_token(self) -> str:
        return "stop"

    @property
    def mask_token(self) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return len(self.TOKENS)

    def tokenize(self, seq: str) -> List[str]:
        return [x for x in seq]

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.VOCAB[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.TOKENS[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return ''.join(tokens)

    def add_special_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        """
        return [self.convert_token_to_id(self.start_token)] + token_ids + \
            [self.convert_token_to_id(self.stop_token)]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def get(name: str) -> TAPETokenizer:
    if name == 'amino_acid':
        return AminoAcidTokenizer()
    elif name == 'unirep':
        return UniRepTokenizer()
    else:
        raise KeyError(f"Unrecognized tokenizer type {name}")
