"""Example of how to add a task in tape_pytorch.

In order to add a new task to TAPE, you must do a few things:

    1) Serialize the data into different splits (e.g. train, val, test) and place
       them in an appropriate folder inside the tape_pytorch data directory. At
       the moment, TAPE supports lmdb and fasta datasets.
    2) Define a dataset as a subclass of TAPEDataset. This should load and return
       the data from your splits.
    2b) Alternatively, as long as you maintain the same API as TAPEDataset, you
        can also simply define a new dataset and load from any arbitrary file
        type.
    3) Define a collate_fn as a method of your dataset which will describe how
       to load in a batch of data (pytorch does not automatically batch variable
       length sequences).
    4) Register the task with TAPE
    5) Register models to the task

This file walks through how to create the 8-class secondary structure prediction
task using the pre-existing secondary structure data.

"""

from typing import Union, List, Tuple, Any, Dict
import torch
from pathlib import Path
import numpy as np

from tape_pytorch.datasets import TAPEDataset
from tape_pytorch import tokenizers
from tape_pytorch.registry import registry
from tape_pytorch import ProteinBertForSequenceToSequenceClassification


# Register the dataset as a new TAPE task. Since it's a classification task
# we need to tell TAPE how many labels the downstream model will use. If this
# wasn't a classification task, that argument could simply be dropped.
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
        """ Override TAPEDataset's __getitem__. The superclass will return
        three things on __getitem__:
            1) The full item loaded from the LMDB/Fasta file (a dictionary)
            2) The tokenized primary sequence
            3) A set of ones - this will be padded into an attention mask
               in the collate_fn
        """

        item, token_ids, input_mask = super().__getitem__(index)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss8'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

        return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        """ Define a collate_fn to convert the variable length sequences into
            a batch of torch tensors. token ids and mask should be padded with
            zeros. Labels for classification should be padded with -1.
        """
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(self.pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(self.pad_sequences(input_mask, 0))
        ss_label = torch.from_numpy(self.pad_sequences(ss_label, -1))

        output = {'input_ids': input_ids,
                  'input_mask': input_mask,
                  'targets': ss_label}

        return output


registry.register_task_model(
    'secondary_structure_8', 'transformer', ProteinBertForSequenceToSequenceClassification)


if __name__ == '__main__':
    """ To actually run the task, you can do one of two things. You can
    simply import the appropriate run function from tape_pytorch.main. The
    possible functions are `run_train`, `run_train_distributed`, and
    `run_eval`. Alternatively, you can add this dataset directly to
    tape_pytorch/datasets.py.
    """
    from tape_pytorch.main import run_train
    run_train()
