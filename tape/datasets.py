from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy
from pathlib import Path
import pickle as pkl
import logging
import random

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from .tokenizers import TAPETokenizer
from .registry import registry

logger = logging.getLogger(__name__)


def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    elif data_file.suffix in {'.fasta', '.fna', '.ffn', '.faa', '.frn'}:
        return FastaDataset(data_file, *args, **kwargs)
    elif data_file.suffix == '.json':
        return JSONDataset(data_file, *args, **kwargs)
    elif data_file.is_dir():
        return NPZDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


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

        # if in_memory:
        cache = list(SeqIO.parse(str(data_file), 'fasta'))
        num_examples = len(cache)
        self._cache = cache
        # else:
            # records = SeqIO.index(str(data_file), 'fasta')
            # num_examples = len(records)
#
            # if num_examples < 10000:
                # logger.info("Reading full fasta file into memory because number of examples "
                            # "is very low. This loads data approximately 20x faster.")
                # in_memory = True
                # cache = list(records.values())
                # self._cache = cache
            # else:
                # self._records = records
                # self._keys = list(records.keys())

        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        # if self._in_memory and self._cache[index] is not None:
        record = self._cache[index]
        # else:
            # key = self._keys[index]
            # record = self._records[key]
            # if self._in_memory:
                # self._cache[index] = record

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
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item


class JSONDataset(Dataset):
    """Creates a dataset from a json file. Assumes that data is
       a JSON serialized list of record, where each record is
       a dictionary.
    Args:
        data_file (Union[str, Path]): Path to json file.
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self, data_file: Union[str, Path], in_memory: bool = True):
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


class NPZDataset(Dataset):
    """Creates a dataset from a directory of npz files.
    Args:
        data_file (Union[str, Path]): Path to directory of npz files
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = True,
                 split_files: Optional[Collection[str]] = None):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)
        file_glob = data_file.glob('*.npz')
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.name in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory")

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_file}")

        self._file_list = file_list

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        item = dict(np.load(self._file_list[index]))
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = self._file_list[index].stem
        return item


@registry.register_task('embed')
class EmbedDataset(Dataset):

    def __init__(self,
                 data_file: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False,
                 convert_tokens_to_ids: bool = True):
        super().__init__()

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.data = dataset_factory(data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return item['id'], token_ids, input_mask

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        ids, tokens, input_mask = zip(*batch)
        ids = list(ids)
        tokens = torch.from_numpy(pad_sequences(tokens))
        input_mask = torch.from_numpy(pad_sequences(input_mask))
        return {'ids': ids, 'input_ids': tokens, 'input_mask': input_mask}  # type: ignore


@registry.register_task('masked_language_modeling')
class MaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'pfam/pfam_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokens = self.tokenizer.tokenize(item['primary'])
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)

        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return masked_token_ids, input_mask, labels, item['clan'], item['family']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_ids, clan, family = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))
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


@registry.register_task('language_modeling')
class LanguageModelingDataset(Dataset):
    """Creates the Language Modeling Pfam Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout'], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'pfam/pfam_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        return token_ids, input_mask, item['clan'], item['family']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, clan, family = tuple(zip(*batch))

        torch_inputs = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        # ignore_index is -1
        torch_labels = torch.from_numpy(pad_sequences(input_ids, -1))
        clan = torch.LongTensor(clan)  # type: ignore
        family = torch.LongTensor(family)  # type: ignore

        return {'input_ids': torch_inputs,
                'input_mask': input_mask,
                'targets': torch_labels}


@registry.register_task('fluorescence')
class FluorescenceDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'fluorescence/fluorescence_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['log_fluorescence'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fluorescence_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore
        fluorescence_true_value = fluorescence_true_value.unsqueeze(1)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fluorescence_true_value}


@registry.register_task('stability')
class StabilityDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'stability/stability_{split}.lmdb'

        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, float(item['stability_score'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, stability_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore
        stability_true_value = stability_true_value.unsqueeze(1)

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': stability_true_value}


@registry.register_task('remote_homology', num_labels=1195)
class RemoteHomologyDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):

        if split not in ('train', 'valid', 'test_fold_holdout',
                         'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'remote_homology/remote_homology_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)
        return token_ids, input_mask, item['fold_label']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, fold_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fold_label = torch.LongTensor(fold_label)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': fold_label}


@registry.register_task('contact_prediction')
class ProteinnetDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):

        if split not in ('train', 'train_unfiltered', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test']")

        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'proteinnet/proteinnet_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        protein_length = len(item['primary'])
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        valid_mask = item['valid_mask']
        contact_map = np.less(squareform(pdist(item['tertiary'])), 8.0).astype(np.int64)

        yind, xind = np.indices(contact_map.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1

        return token_ids, input_mask, contact_map, protein_length

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, contact_labels, protein_length = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        contact_labels = torch.from_numpy(pad_sequences(contact_labels, -1))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': contact_labels,
                'protein_length': protein_length}


@registry.register_task('secondary_structure', num_labels=3)
class SecondaryStructureDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False):

        if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        token_ids = self.tokenizer.encode(item['primary'])
        input_mask = np.ones_like(token_ids)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss3'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(pad_sequences(ss_label, -1))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output


@registry.register_task('trrosetta')
class TRRosettaDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, TAPETokenizer] = 'iupac',
                 in_memory: bool = False,
                 max_seqlen: int = 300):
        if split not in ('train', 'valid'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid']")
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_path = data_path / 'trrosetta'
        split_files = (data_path / f'{split}_files.txt').read_text().split()
        self.data = NPZDataset(data_path / 'npz', in_memory, split_files=split_files)

        self._dist_bins = np.arange(2, 20.1, 0.5)
        self._dihedral_bins = (15 + np.arange(-180, 180, 15)) / 180 * np.pi
        self._planar_bins = (15 + np.arange(0, 180, 15)) / 180 * np.pi
        self._split = split
        self.max_seqlen = max_seqlen
        self.msa_cutoff = 0.8
        self.penalty_coeff = 4.5

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        msa = item['msa']
        dist = item['dist6d']
        omega = item['omega6d']
        theta = item['theta6d']
        phi = item['phi6d']

        if self._split == 'train':
            msa = self._subsample_msa(msa)
        elif self._split == 'valid':
            msa = msa[:20000]  # runs out of memory if msa is way too big
        msa, dist, omega, theta, phi = self._slice_long_sequences(
            msa, dist, omega, theta, phi)

        mask = dist == 0

        dist_bins = np.digitize(dist, self._dist_bins)
        omega_bins = np.digitize(omega, self._dihedral_bins) + 1
        theta_bins = np.digitize(theta, self._dihedral_bins) + 1
        phi_bins = np.digitize(phi, self._planar_bins) + 1

        dist_bins[mask] = 0
        omega_bins[mask] = 0
        theta_bins[mask] = 0
        phi_bins[mask] = 0

        dist_bins[np.diag_indices_from(dist_bins)] = -1

        # input_mask = np.ones_like(msa[0])

        return msa, dist_bins, omega_bins, theta_bins, phi_bins

    def _slice_long_sequences(self, msa, dist, omega, theta, phi):
        seqlen = msa.shape[1]
        if self.max_seqlen > 0 and seqlen > self.max_seqlen:
            start = np.random.randint(seqlen - self.max_seqlen + 1)
            end = start + self.max_seqlen

            msa = msa[:, start:end]
            dist = dist[start:end, start:end]
            omega = omega[start:end, start:end]
            theta = theta[start:end, start:end]
            phi = phi[start:end, start:end]

        return msa, dist, omega, theta, phi

    def _subsample_msa(self, msa):
        num_alignments, seqlen = msa.shape

        if num_alignments < 10:
            return msa

        num_sample = int(10 ** np.random.uniform(np.log10(num_alignments)) - 10)

        if num_sample <= 0:
            return msa[0][None, :]
        elif num_sample > 20000:
            num_sample = 20000

        indices = np.random.choice(
            msa.shape[0] - 1, size=num_sample, replace=False) + 1
        indices = np.pad(indices, [1, 0], 'constant')  # add the sequence back in
        return msa[indices]

    def collate_fn(self, batch):
        msa, dist_bins, omega_bins, theta_bins, phi_bins = tuple(zip(*batch))
        # features = pad_sequences([self.featurize(msa_) for msa_ in msa], 0)
        msa1hot = pad_sequences(
            [F.one_hot(torch.LongTensor(msa_), 21) for msa_ in msa], 0, torch.float)
        # input_mask = torch.FloatTensor(pad_sequences(input_mask, 0))
        dist_bins = torch.LongTensor(pad_sequences(dist_bins, -1))
        omega_bins = torch.LongTensor(pad_sequences(omega_bins, 0))
        theta_bins = torch.LongTensor(pad_sequences(theta_bins, 0))
        phi_bins = torch.LongTensor(pad_sequences(phi_bins, 0))

        return {'msa1hot': msa1hot,
                # 'input_mask': input_mask,
                'dist': dist_bins,
                'omega': omega_bins,
                'theta': theta_bins,
                'phi': phi_bins}

    def featurize(self, msa):
        msa = torch.LongTensor(msa)
        msa1hot = F.one_hot(msa, 21).float()

        seqlen = msa1hot.size(1)

        weights = self.reweight(msa1hot)
        features_1d = self.extract_features_1d(msa1hot, weights)
        features_2d = self.extract_features_2d(msa1hot, weights)

        features = torch.cat((
            features_1d.unsqueeze(1).repeat(1, seqlen, 1),
            features_1d.unsqueeze(0).repeat(seqlen, 1, 1),
            features_2d), -1)

        features = features.permute(2, 0, 1)

        return features

    def reweight(self, msa1hot):
        # Reweight
        seqlen = msa1hot.size(1)
        id_min = seqlen * self.msa_cutoff
        id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        weights = 1.0 / id_mask.float().sum(-1)
        return weights

    def extract_features_1d(self, msa1hot, weights):
        # 1D Features
        seqlen = msa1hot.size(1)
        f1d_seq = msa1hot[0, :, :20]

        # msa2pssm
        beff = weights.sum()
        f_i = (weights[:, None, None] * msa1hot).sum(0) / beff + 1e-9
        h_i = (-f_i * f_i.log()).sum(1, keepdims=True)
        f1d_pssm = torch.cat((f_i, h_i), dim=1)

        f1d = torch.cat((f1d_seq, f1d_pssm), dim=1)
        f1d = f1d.view(seqlen, 42)
        return f1d

    def extract_features_2d(self, msa1hot, weights):
        # 2D Features
        num_alignments = msa1hot.size(0)
        seqlen = msa1hot.size(1)
        num_symbols = 21
        if num_alignments == 1:
            # No alignments, predict from sequence alone
            f2d_dca = torch.zeros(seqlen, seqlen, 442, dtype=torch.float)
        else:
            # fast_dca

            # covariance
            x = msa1hot.view(num_alignments, seqlen * num_symbols)
            num_points = weights.sum() - weights.mean().sqrt()
            mean = (x * weights[:, None]).sum(0, keepdims=True) / num_points
            x = (x - mean) * weights[:, None].sqrt()
            cov = torch.matmul(x.transpose(-1, -2), x) / num_points

            # inverse covariance
            reg = torch.eye(seqlen * num_symbols) * self.penalty_coeff / weights.sum().sqrt()
            cov_reg = cov + reg
            inv_cov = torch.inverse(cov_reg)

            x1 = inv_cov.view(seqlen, num_symbols, seqlen, num_symbols)
            x2 = x1.permute(0, 2, 1, 3)
            features = x2.reshape(seqlen, seqlen, num_symbols * num_symbols)

            x3 = (x1[:, :-1, :, :-1] ** 2).sum((1, 3)).sqrt() * (1 - torch.eye(seqlen))
            apc = x3.sum(0, keepdims=True) * x3.sum(1, keepdims=True) / x3.sum()
            contacts = (x3 - apc) * (1 - torch.eye(seqlen))

            f2d_dca = torch.cat([features, contacts[:, :, None]], axis=2)

        return f2d_dca
