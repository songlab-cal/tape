from typing import Union, List, Optional
from pathlib import Path
import logging

import sentencepiece as spm

logger = logging.getLogger(__name__)


class PfamTokenizer(object):
    r"""
    Constructs a PfamTokenizer.

    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_wordpiece_only=False
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_wordpiece_only=False

    Command To Train:
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
    def from_pretrained(cls, model_file: Union[str, Path], **kwargs):
        return cls(model_file=model_file, **kwargs)
