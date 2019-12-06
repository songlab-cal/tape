from typing import Union, List, Tuple, Sequence, Dict, Any
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from .tokenizers import TAPETokenizer
from .registry import registry

logger = logging.getLogger(__name__)


class FastaDataset(Dataset):
    """Creates a dataset from a fasta file.
    Args:
        data_file (Union[str, Path]): Path to fasta file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        from Bio import SeqIO
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        if in_memory:
            cache = list(SeqIO.parse(str(data_file)))
            num_examples = len(cache)
            self._cache = cache
        else:
            records = SeqIO.index(str(data_file), 'fasta')
            num_examples = len(records)

            if num_examples < 10000:
                logger.info("Reading full fasta file into memory because number of examples "
                            "is very low. This loads data approximately 20x faster.")
                in_memory = True
                cache = list(records.values())
                self._cache = cache
            else:
                self._records = records
                self._keys = list(records.keys())

        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            record = self._cache[index]
        else:
            key = self._keys[index]
            record = self._records[key]
            if self._in_memory:
                self._cache[index] = record

        item = {'id': record.id,
                'primary': str(record.seq),
                'protein_length': len(record.seq)}
        return item


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
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
        if 'id' not in item:
            item['id'] = str(index)
        return item


class JSONDataset(Dataset):
    """Creates a dataset from a json file. Assumes that data is
       a JSON serialized list of record, where each record is
       a dictionary.
    Args:
        data_file (Union[str, Path]): Path to json file.
    """

    def __init__(self, data_file: Union[str, Path]):
        import json
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        records = json.loads(data_file.read_text())

        if not isinstance(records, list):
            raise TypeError(f"TAPE JSONDataset requires a json serialized list, "
                            f"received {type(records)}")
        self._records = records
        self._num_examples = len(records)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        item = self._records[index]
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = str(index)
        return item


@registry.register_task('embed')
class TAPEDataset(Dataset):

    def __init__(self,
                 data_file: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'amino_acid',
                 in_memory: bool = False,
                 convert_tokens_to_ids: bool = True):
        super().__init__()

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)

        assert isinstance(tokenizer, TAPETokenizer)
        self.tokenizer = tokenizer
        self._convert_tokens_to_ids = convert_tokens_to_ids

        if data_file.suffix == '.lmdb':
            self._dataset: Dataset = LMDBDataset(data_file, in_memory)
        elif data_file.suffix == '.fasta':
            self._dataset = FastaDataset(data_file, in_memory)
        elif data_file.suffix == '.json':
            self._dataset = JSONDataset(data_file)
        else:
            raise ValueError(f"Unrecognized dataset type: {data_file.suffix}")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        item = self._dataset[index]
        tokens = self.tokenizer.tokenize(item['primary'])
        tokens = [self.tokenizer.start_token] + tokens + [self.tokenizer.stop_token]

        if self._convert_tokens_to_ids:
            tokens = np.array(self.tokenizer.convert_tokens_to_ids(tokens), np.int64)

        input_mask = np.ones([len(tokens)], dtype=np.int64)

        return item, tokens, input_mask

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        items, tokens, input_mask = zip(*batch)
        items = list(items)
        tokens = torch.from_numpy(self.pad_sequences(tokens))
        input_mask = torch.from_numpy(self.pad_sequences(input_mask))
        return {'input_ids': tokens, 'input_mask': input_mask}

    def pad_sequences(self, sequences: Sequence[np.ndarray], constant_value=0) -> np.ndarray:
        batch_size = len(sequences)
        shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()
        array = np.zeros(shape, sequences[0].dtype) + constant_value

        for arr, seq in zip(array, sequences):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq

        return array


@registry.register_task('language_modeling')
class PfamDataset(TAPEDataset):
    """Creates the Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        mode (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Union[str, TAPETokenizer] = 'amino_acid',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized mode: {mode}. "
                f"Must be one of ['train', 'valid', 'holdout']")

        data_path = Path(data_path)
        data_file = f'pfam/pfam_{mode}.lmdb'

        super().__init__(
            data_path / data_file, tokenizer, in_memory, convert_tokens_to_ids=False)

    def __getitem__(self, index):
        item, tokens, input_mask = super().__getitem__(index)

        masked_tokens, labels = self._apply_bert_mask(tokens)

        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return masked_token_ids, input_mask, labels, item['clan'], item['family']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_ids, clan, family = tuple(zip(*batch))

        input_ids = torch.from_numpy(self.pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(self.pad_sequences(input_mask, 0))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(self.pad_sequences(lm_label_ids, -1))
        clan = torch.LongTensor(clan)  # type: ignore
        family = torch.LongTensor(family)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_ids}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
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


@registry.register_task('fluorescence')
class FluorescenceDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Union[str, TAPETokenizer] = 'amino_acid',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized mode: {mode}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'fluorescence/fluorescence_{mode}.lmdb'

        super().__init__(
            data_path / data_file, tokenizer, in_memory, convert_tokens_to_ids=True)

    def __getitem__(self, index: int):
        item, token_ids, input_mask = super().__getitem__(index)
        return token_ids, input_mask, float(item['log_fluorescence'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fluorescence_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(self.pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(self.pad_sequences(input_mask, 0))
        fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore
        fluorescence_true_value = fluorescence_true_value.unsqueeze(1)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fluorescence_true_value}


@registry.register_task('stability')
class StabilityDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Union[str, TAPETokenizer] = 'amino_acid',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized mode: {mode}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'stability/stability_{mode}.lmdb'

        super().__init__(
            data_path / data_file, tokenizer, in_memory, convert_tokens_to_ids=True)

    def __getitem__(self, index: int):
        item, token_ids, input_mask = super().__getitem__(index)
        return token_ids, input_mask, float(item['stability_score'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, stability_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(self.pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(self.pad_sequences(input_mask, 0))
        stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore
        stability_true_value = stability_true_value.unsqueeze(1)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': stability_true_value}


@registry.register_task('remote_homology', num_labels=1195)
class RemoteHomologyDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Union[str, TAPETokenizer] = 'amino_acid',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'test_fold_holdout',
                        'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")

        data_path = Path(data_path)
        data_file = f'remote_homology/remote_homology_{mode}.lmdb'

        super().__init__(
            data_path / data_file, tokenizer, in_memory, convert_tokens_to_ids=True)

    def __getitem__(self, index: int):
        item, token_ids, input_mask = super().__getitem__(index)
        return token_ids, input_mask, item['fold_label']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fold_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(self.pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(self.pad_sequences(input_mask, 0))
        fold_label = torch.LongTensor(fold_label)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fold_label}


@registry.register_task('contact_prediction')
class ProteinnetDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Union[str, TAPETokenizer] = 'amino_acid',
                 in_memory: bool = False):

        if mode not in ('train', 'train_unfiltered', 'valid', 'test'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'proteinnet/proteinnet_{mode}.lmdb'
        self._mode = mode
        super().__init__(
            data_path / data_file, tokenizer, in_memory, convert_tokens_to_ids=True)

    def __getitem__(self, index: int):
        item, token_ids, input_mask = super().__getitem__(index)

        valid_mask = item['valid_mask']
        contact_map = np.less(squareform(pdist(item['tertiary'])), 8.0).astype(np.int64)

        yind, xind = np.indices(contact_map.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1

        return token_ids, input_mask, contact_map

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, contact_labels = tuple(zip(*batch))
        input_ids = torch.from_numpy(self.pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(self.pad_sequences(input_mask, 0))
        contact_labels = torch.from_numpy(self.pad_sequences(contact_labels, -1))

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': contact_labels}


@registry.register_task('secondary_structure', num_labels=3)
class SecondaryStructureDataset(TAPEDataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Union[str, TAPETokenizer] = 'amino_acid',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")

        data_path = Path(data_path)
        data_file = f'secondary_structure/secondary_structure_{mode}.lmdb'
        super().__init__(
            data_path / data_file, tokenizer, in_memory)

    def __getitem__(self, index: int):
        item, token_ids, input_mask = super().__getitem__(index)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss3'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(self.pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(self.pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(self.pad_sequences(ss_label, -1))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output
