from typing import Union, List, Tuple, Optional, Sequence
from pathlib import Path
import pickle as pkl
import logging
import random

import lmdb
import sentencepiece as spm
import numpy as np
from torch.utils.data import Dataset

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


class LMDBDataset(Dataset):
    """Creates the Pfam Dataset
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = len(pkl.loads(txn.get(b'keys')))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if self._in_memory:
                    self._cache[index] = item
        return item


class PfamDataset(LMDBDataset):
    """Creates the Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        mode (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Optional[PfamTokenizer] = None,
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'holdout'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of ['train', 'valid', 'holdout']")

        data_path = Path(data_path)
        data_file = data_path / 'pfam' / f'pfam_{mode}.lmdb'

        if tokenizer is None:
            model_file = data_path / 'pfam.model'
            if not (model_file.exists()):
                raise FileNotFoundError(
                    "PfamDataset requires a tokenizer. If tokenizer is not provided it "
                    "looks for files in data_path/pfam.model. You must either place the "
                    "model file there or provide the tokenizer yourself.")
            tokenizer = PfamTokenizer.from_pretrained(model_file)

        preprocess_function = BertPreprocessBatch(tokenizer)
        self.preprocess = preprocess_function
        super().__init__(data_file, in_memory)

    def __getitem__(self, index):
        return self.preprocess(super().__getitem__(index))


class PfamBatch:

    def __init__(self, batch):
        input_ids, input_mask, lm_label_ids, clan, family = tuple(zip(*batch))

        input_ids = self._pad_numpy(input_ids, 0)  # pad input_ids with zeros
        input_mask = self._pad_numpy(input_mask, 0)  # pad input_mask with zeros
        lm_label_ids = self._pad_numpy(lm_label_ids, -1)  # pad lm_label_ids with minus ones
        clan = np.stack(clan, 0)
        family = np.stack(family, 0)

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.lm_label_ids = lm_label_ids
        self.clan = clan
        self.family = family

    def _pad_numpy(self, sequences: Sequence[np.ndarray], constant_value=0) -> np.ndarray:
        batch_size = len(sequences)
        shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
        array = np.zeros(shape, sequences[0].dtype) + constant_value

        for arr, seq in zip(array, sequences):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq

        return array


class BertPreprocessBatch(object):

    def __init__(self, tokenizer: PfamTokenizer):

        self.tokenizer = tokenizer

    def __call__(self, data):
        tokenize_primary = self.tokenizer.tokenize(data['primary'])

        print(len(data['primary']), len(tokenize_primary))

        # transform sample to features
        input_ids, input_mask, lm_label_ids = self.convert_example_to_features(
            tokenize_primary, self.tokenizer)

        tensors = (
            input_ids,
            input_mask,
            lm_label_ids,
            data['clan'],
            data['family'])

        return tensors

    def convert_example_to_features(self, primary, tokenizer):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        primary, primary_label = self.random_word(primary, tokenizer)
        # concatenate lm labels and account for CLS, SEP
        # lm_label_ids = ([-1] + primary_label + [-1])
        lm_label_ids = [-1] + primary_label + [-1]
        # image_label = ([-1] + image_label)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []

        tokens.append(tokenizer.cls_token)

        tokens = [tokenizer.cls_token] + primary + [tokenizer.sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))

        # Zero-pad up to the sequence length.
        # while len(input_ids) < max_seq_length:
            # input_ids.append(0)
            # input_mask.append(0)
            # lm_label_ids.append(-1)

        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(lm_label_ids) == max_seq_length

        return input_ids, input_mask, lm_label_ids

    def _bert_mask_token(self, token: str, tokenizer: PfamTokenizer) -> Tuple[str, int]:
        prob = random.random()

        if prob < 0.15:
            prob /= 0.15
            label = tokenizer.convert_token_to_id(token)

            if prob < 0.8:
                # 80% random change to mask token
                token = tokenizer.mask_token
            elif prob < 0.9:
                # 10% chance to change to random token
                token = tokenizer.convert_id_to_token(random.randint(0, tokenizer.vocab_size))
            else:
                # 10% chance to keep current token
                pass
        else:
            label = -1

        return token, label

    def random_word(self, tokens: List[str], tokenizer: PfamTokenizer) -> Tuple[List[str], List[int]]:
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        return list(map(list, zip(*(self._bert_mask_token(token, tokenizer) for token in tokens))))  # type: ignore
