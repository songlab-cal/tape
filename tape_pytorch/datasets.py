from typing import Union, List, Tuple, Optional, Sequence, Dict, Any
from copy import copy
from abc import ABC, abstractmethod
from pathlib import Path
import pickle as pkl
import logging
import random

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

import tape_pytorch.tokenizers as tokenizers

logger = logging.getLogger(__name__)


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
            num_examples = pkl.loads(txn.get(b'num_examples'))

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


class PaddedBatch(ABC):

    @abstractmethod
    def __call__(self, batch: List[Sequence[np.ndarray]]) -> Tuple[np.ndarray]:
        return NotImplemented

    def _pad(self, sequences: Sequence[np.ndarray], constant_value=0) -> torch.Tensor:
        batch_size = len(sequences)
        shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
        array = np.zeros(shape, sequences[0].dtype) + constant_value

        for arr, seq in zip(array, sequences):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq

        return torch.from_numpy(array)


class TAPEDataset(LMDBDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 data_file: Union[str, Path],
                 tokenizer: Optional[Union[str, tokenizers.TAPETokenizer]] = 'pfam',
                 in_memory: bool = False,
                 convert_tokens_to_ids: bool = True):

        data_path = Path(data_path)

        if isinstance(tokenizer, str):
            model_file = data_path / 'pfam.model'
            tokenizer = tokenizers.get(tokenizer, model_file)

        self.tokenizer = tokenizer
        self._convert_tokens_to_ids = convert_tokens_to_ids
        super().__init__(data_path / data_file, in_memory)

    def __getitem__(self, index: int) -> Tuple[Dict[str, Any], Union[List[int], List[str]], List[int]]:
        item = super().__getitem__(index)
        tokens = self.tokenizer.tokenize(item['primary'])
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]

        if self._convert_tokens_to_ids:
            tokens = np.array(self.tokenizer.convert_tokens_to_ids(tokens), np.int64)

        attention_mask = np.ones([len(tokens)], dtype=np.int64)

        return item, tokens, attention_mask


class PfamDataset(TAPEDataset):
    """Creates the Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        mode (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Optional[Union[str, tokenizers.TAPETokenizer]] = 'pfam',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'holdout'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of ['train', 'valid', 'holdout']")

        data_path = Path(data_path)
        data_file = f'pfam/pfam_{mode}.lmdb'

        super().__init__(data_path, data_file, tokenizer, in_memory, convert_tokens_to_ids=False)

    def __getitem__(self, index):
        item, tokens, attention_mask = super().__getitem__(index)

        masked_tokens, labels = self._apply_bert_mask(tokens)

        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return masked_token_ids, attention_mask, labels, item['clan'], item['family']

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with cls_token and sep_token, ignore these
            if token in (self.tokenizer.cls_token, self.tokenizer.sep_token):
                pass

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels


class PfamBatch(PaddedBatch):

    def __call__(self, batch):
        input_ids, input_mask, lm_label_ids, clan, family = tuple(zip(*batch))

        input_ids = self._pad(input_ids, 0)  # pad input_ids with zeros
        input_mask = self._pad(input_mask, 0)  # pad input_mask with zeros
        lm_label_ids = self._pad(lm_label_ids, -1)  # pad lm_label_ids with minus ones
        clan = torch.LongTensor(clan)
        family = torch.LongTensor(family)

        return input_ids, input_mask, lm_label_ids, clan, family


class FluorescenceDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Optional[Union[str, tokenizers.TAPETokenizer]] = 'pfam',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'fluorescence/fluorescence_{mode}.lmdb'

        super().__init__(data_path, data_file, tokenizer, in_memory, convert_tokens_to_ids=True)

    def __getitem__(self, index: int):
        item, token_ids, attention_mask = super().__getitem__(index)
        return token_ids, attention_mask, float(item['log_fluorescence'][0])


class FluorescenceBatch(PaddedBatch):
    pass


class StabilityDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Optional[Union[str, tokenizers.TAPETokenizer]] = 'pfam',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'stability/stability_{mode}.lmdb'

        super().__init__(data_path, data_file, tokenizer, in_memory, convert_tokens_to_ids=True)

    def __getitem__(self, index: int):
        item, token_ids, attention_mask = super().__getitem__(index)
        return token_ids, attention_mask, float(item['stability_score'][0])


class StabilityBatch(PaddedBatch):
    pass


class RemoteHomologyDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Optional[Union[str, tokenizers.TAPETokenizer]] = 'pfam',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")

        data_path = Path(data_path)
        data_file = f'remote_homology/remote_homology_{mode}.lmdb'

        super().__init__(data_path, data_file, tokenizer, in_memory, convert_tokens_to_ids=True)

    def __getitem__(self, index: int):
        item, token_ids, attention_mask = super().__getitem__(index)
        return token_ids, attention_mask, item['fold_label']


class RemoteHomologyBatch(PaddedBatch):
    pass


class ProteinnetDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Optional[Union[str, tokenizers.TAPETokenizer]] = 'pfam',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of "
                             f"['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'proteinnet/proteinnet_{mode}.lmdb'
        super().__init__(data_path, data_file, tokenizer, in_memory, convert_tokens_to_ids=True)

    def __getitem__(self, index: int):
        item, token_ids, attention_mask = super().__getitem__(index)

        valid_mask = item['valid_mask']
        contact_map = squareform(pdist(item['tertiary'])) < 8.0

        return token_ids, attention_mask, contact_map, valid_mask


class ProteinnetBatch(PaddedBatch):
    pass


class SecondaryStructureDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Optional[Union[str, tokenizers.TAPETokenizer]] = 'pfam',
                 in_memory: bool = False,
                 num_classes: int = 3):

        if mode not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")

        data_path = Path(data_path)
        data_file = f'secondary_structure/secondary_structure_{mode}.lmdb'
        super().__init__(data_path, data_file, tokenizer, in_memory, convert_tokens_to_ids=True)

        self._num_classes = num_classes

    def __getitem__(self, index: int):
        item, token_ids, attention_mask = super().__getitem__(index)
        return token_ids, attention_mask, item[f'ss{self._num_classes}']


class SecondaryStructureBatch(PaddedBatch):
    pass
