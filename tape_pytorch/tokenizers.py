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
        cls_token = [self.convert_token_to_id(self.cls_token)]
        sep_token = [self.convert_token_to_id(self.sep_token)]
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


@registry.register_tokenizer('bpe')
class BPETokenizer(TAPETokenizer):
    r"""
    Constructs a BPETokenizer.

    Args:
        corpus_file (Union[str, Path], optional): Path to a full corpus file for Pfam.
            Must provide this or model_file. If this argument is provided, it trains
            the tokenizer.
        model_file (Union[str, Path], optional): Path to a trained model file. Must
            provide this or corpus_file.

    Command To Train (This is run if a corpus file is passed in):
        spm_train --input pfam_strings.txt \
                  --model_prefix pfam \
                  --vocab_size 8000 \
                  --bos_piece <cls> \
                  --eos_piece <sep> \
                  --user_defined_symbols=<mask> \
                  --pad_id 0 \
                  --bos_id 2 \
                  --eos_id 3 \
                  --unk_id 4 \
                  --input_sentence_size=1000000 \
                  --shuffle_input_sentence=true \
                  --model_type=bpe
    """

    def __init__(self, *,  # only accept keyword arguments
                 corpus_file: Optional[Union[str, Path]] = None,
                 model_file: Optional[Union[str, Path]] = None,
                 unk_token: str = "<unk>",
                 sep_token: str = "<sep>",
                 pad_token: str = "<pad>",
                 cls_token: str = "<cls>",
                 mask_token: str = "<mask>"):

        corpus_file_provided = corpus_file is not None
        model_file_provided = model_file is not None

        if not (corpus_file_provided ^ model_file_provided):
            logger.error("Must provide either a corpus file or a trained model file")
            raise ValueError("Must provide either a corpus file or a trained model file")

        if corpus_file_provided:
            logger.info(f'Corpus file is provided, running sentencepiece training')
            command = [f'--input={corpus_file}',
                       f'--model_prefix=pfam',
                       f'--vocab_size=8000',
                       f'--bos_piece={cls_token}',
                       f'--eos_piece={sep_token}',
                       f'--user_defined_symbols={mask_token}',
                       f'--pad_id=0',
                       f'--bos_id=2',
                       f'--eos_id=3',
                       f'--unk_id=4',
                       f'--input_sentence_size=1000000',
                       f'--shuffle_input_sentence=true',
                       f'--model_type=bpe',
                       f'--character_coverage=1.0']
            spm.SentencePieceTrainer.Train(' '.join(command))
            model_file = 'pfam.model'

        assert model_file is not None
        logger.info(f'Loading sentencepiece model from {model_file}')
        model_file = Path(model_file)

        if not model_file.is_file():
            raise FileNotFoundError(model_file)
        sp = spm.SentencePieceProcessor()
        sp.Load(str(model_file))

        self.sp = sp

        super().__init__(unk_token, sep_token, pad_token, cls_token, mask_token)

    @property
    def vocab_size(self) -> int:
        return len(self.sp)

    def tokenize(self, text: str) -> List[str]:
        return self.sp.encode_as_pieces(text)

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.sp.piece_to_id(token)

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.sp.id_to_piece(index)

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return self.sp.decode_pieces(tokens)

    def add_special_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """

        return [self.convert_token_to_id(self.cls_token)] + token_ids + \
            [self.convert_token_to_id(self.sep_token)]

    @classmethod
    def from_pretrained(cls, model_file: Optional[Union[str, Path]] = None, **kwargs):
        if model_file is None:
            raise ValueError("Must specify pretrained model file")
        return cls(model_file=model_file, **kwargs)


def get(name: str) -> Type[TAPETokenizer]:
    if name == 'amino_acid':
        return AminoAcidTokenizer
    elif name == 'unirep':
        return UniRepTokenizer
    elif name == 'bpe':
        return BPETokenizer
    else:
        raise KeyError(f"Unrecognized tokenizer type {name}")
