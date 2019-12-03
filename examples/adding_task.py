"""Example of how to add a task in tape_pytorch.
"""

from typing import Union, List, Tuple, Any, Dict
import torch
from pathlib import Path
import numpy as np

from tape_pytorch.datasets import TAPEDataset
from tape_pytorch import tokenizers
from tape_pytorch.registry import registry
from tape_pytorch import ProteinBertForSequenceToSequenceClassification


@registry.register_task('secondary_structure_8', num_labels=8)
class SecondaryStructure8ClassDataset(TAPEDataset):
    """ Defines the 8-class secondary structure prediction dataset.

    Args:
        data_path (Union[str, Path]): Path to tape data directory. By default, this is
            assumed to be `./data`. Can be altered on the command line with the --data_dir
            flag.
        mode (str): The specific dataset split to load often <train, valid, test>. In the
            case of secondary structure, there are three test datasets so each of these
            has a separate mode flag.
        tokenizer (str): The model tokenizer to use when returning tokenized indices.
        in_memory (bool): Whether to load the entire dataset into memory or to keep
            it on disk.
    """

    def __init__(self,
                 data_path: Union[str, Path],
                 mode: str,
                 tokenizer: Union[str, tokenizers.TAPETokenizer] = 'amino_acid',
                 in_memory: bool = False):

        if mode not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized mode: {mode}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")

        data_path = Path(data_path)
        data_file = f'secondary_structure/secondary_structure_{mode}.lmdb'
        super().__init__(data_path, data_file, tokenizer, in_memory)

    def __getitem__(self, index: int):
        item, token_ids, input_mask = super().__getitem__(index)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss8'], np.int64)
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


registry.register_task_model('secondary_structure_8', 'transformer', ProteinBertForSequenceToSequenceClassification)


if __name__ == '__main__':
    from tape_pytorch.main import run_train
    run_train()
