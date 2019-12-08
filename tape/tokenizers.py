from typing import List
import logging
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)

IUPAC_CODES = OrderedDict([
    ('Ala', 'A'),
    ('Asx', 'B'),
    ('Cys', 'C'),
    ('Asp', 'D'),
    ('Glu', 'E'),
    ('Phe', 'F'),
    ('Gly', 'G'),
    ('His', 'H'),
    ('Ile', 'I'),
    ('Lys', 'K'),
    ('Leu', 'L'),
    ('Met', 'M'),
    ('Asn', 'N'),
    ('Pro', 'P'),
    ('Gln', 'Q'),
    ('Arg', 'R'),
    ('Ser', 'S'),
    ('Thr', 'T'),
    ('Sec', 'U'),
    ('Val', 'V'),
    ('Trp', 'W'),
    ('Xaa', 'X'),
    ('Tyr', 'Y'),
    ('Glx', 'Z')])

IUPAC_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("<mask>", 1),
    ("<cls>", 2),
    ("<sep>", 3),
    ("<unk>", 4),
    ("A", 5),
    ("B", 6),
    ("C", 7),
    ("D", 8),
    ("E", 9),
    ("F", 10),
    ("G", 11),
    ("H", 12),
    ("I", 13),
    ("K", 14),
    ("L", 15),
    ("M", 16),
    ("N", 17),
    ("O", 18),
    ("P", 19),
    ("Q", 20),
    ("R", 21),
    ("S", 22),
    ("T", 23),
    ("U", 24),
    ("V", 25),
    ("W", 26),
    ("X", 27),
    ("Y", 28),
    ("Z", 29)])

UNIREP_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("M", 1),
    ("R", 2),
    ("H", 3),
    ("K", 4),
    ("D", 5),
    ("E", 6),
    ("S", 7),
    ("T", 8),
    ("N", 9),
    ("Q", 10),
    ("C", 11),
    ("U", 12),
    ("G", 13),
    ("P", 14),
    ("A", 15),
    ("V", 16),
    ("I", 17),
    ("F", 18),
    ("Y", 19),
    ("W", 20),
    ("L", 21),
    ("O", 22),
    ("X", 23),
    ("Z", 23),
    ("B", 23),
    ("J", 23),
    ("<cls>", 24),
    ("<sep>", 25)])


class TAPETokenizer():
    r"""TAPE Tokenizer. Can use different vocabs depending on the model.
    """

    def __init__(self, vocab: str = 'iupac'):
        if vocab == 'iupac':
            self.vocab = IUPAC_VOCAB
        elif vocab == 'unirep':
            self.vocab = UNIREP_VOCAB
        self.tokens = list(self.vocab.keys())
        self._vocab_type = vocab
        assert self.start_token in self.vocab and self.stop_token in self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        return "<cls>"

    @property
    def stop_token(self) -> str:
        return "<sep>"

    @property
    def mask_token(self) -> str:
        if "<mask>" in self.vocab:
            return "<mask>"
        else:
            raise RuntimeError(f"{self._vocab_type} vocab does not support masking")

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.tokens[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.convert_id_to_token(id_) for id_ in indices]

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return ''.join(tokens)

    def add_special_tokens(self, token_ids: List[str]) -> List[str]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        cls_token = [self.start_token]
        sep_token = [self.stop_token]
        return cls_token + token_ids + sep_token

    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenize(text)
        tokens = self.add_special_tokens(tokens)
        token_ids = self.convert_tokens_to_ids(tokens)
        return np.array(token_ids, np.int64)

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls()
