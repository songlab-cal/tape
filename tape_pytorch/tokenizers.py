from typing import Union, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod, abstractproperty
import logging

import sentencepiece as spm

logger = logging.getLogger(__name__)


class TAPETokenizer(ABC):
    r"""
    Abstract Tokenizer Class
    """

    @abstractproperty
    def vocab_size(self) -> int:
        return len(DummyTokenizer.TOKENS)

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        return NotImplemented

    @abstractmethod
    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        return NotImplemented

    @abstractmethod
    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return NotImplemented

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return NotImplemented

    @abstractmethod
    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return NotImplemented

    @abstractmethod
    def add_special_tokens_single_sentence(self, token_ids: List[int]) -> List[int]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        return NotImplemented

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        raise NotImplementedError("Can't do this for Pfam")

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls


class DummyTokenizer(TAPETokenizer):
    r"""
    Constructs a DummyTokenizer
    """

    TOKENS = ["<pad>", "<mask>", "<cls>", "<sep>", "<unk>",
              "A", "B", "C", "D", "E", "F", "G", "H", "I",
              "K", "L", "M", "N", "O", "P", "Q", "R", "S",
              "T", "U", "V", "W", "X", "Y", "Z"]
    VOCAB = {token: i for i, token in enumerate(TOKENS)}

    def __init__(self, *,  # only accept keyword arguments
                 unk_token: str = "<unk>",
                 sep_token: str = "<sep>",
                 pad_token: str = "<pad>",
                 cls_token: str = "<cls>",
                 mask_token: str = "<mask>"):

        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

    @property
    def vocab_size(self) -> int:
        return len(DummyTokenizer.TOKENS)

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return DummyTokenizer.VOCAB[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return DummyTokenizer.TOKENS[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return ''.join(tokens)

    def add_special_tokens_single_sentence(self, token_ids: List[int]) -> List[int]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """

        return [self.convert_token_to_id(self.cls_token)] + token_ids + [self.convert_token_to_id(self.sep_token)]

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        raise NotImplementedError("Can't do this for Pfam")

    @classmethod
    def from_pretrained(cls, **kwargs):
        return cls


class PfamTokenizer(TAPETokenizer):
    r"""
    Constructs a PfamTokenizer.

    Args:
        corpus_file (Union[str, Path], optional): Path to a full corpus file for Pfam. Must provide this or model_file.
            If this argument is provided, it trains the tokenizer.
        model_file (Union[str, Path], optional): Path to a trained model file. Must provide this or corpus_file.

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
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

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

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return self.sp.decode_pieces(tokens)

    def add_special_tokens_single_sentence(self, token_ids: List[int]) -> List[int]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """

        return [self.convert_token_to_id(self.cls_token)] + token_ids + [self.convert_token_to_id(self.sep_token)]

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A BERT sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        raise NotImplementedError("Can't do this for Pfam")

    @classmethod
    def from_pretrained(cls, model_file: Optional[Union[str, Path]] = None, **kwargs):
        if model_file is None:
            raise ValueError("Must specify pretrained model file")
        return cls(model_file=model_file, **kwargs)


def get(tokenizer_type: str, model_file: Optional[Union[str, Path]] = None) -> TAPETokenizer:
    if tokenizer_type == 'dummy':
        return DummyTokenizer()
    elif tokenizer_type == 'pfam':
        if model_file is None:
            raise ValueError("Must provide a pretrained model file when creating a PfamTokenizer")
        return PfamTokenizer(model_file=model_file)
    else:
        raise ValueError(f"Unrecognized tokenizer: {tokenizer_type}")
