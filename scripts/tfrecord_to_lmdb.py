
from typing import Union, List
import lmdb
import os
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="numpy")
from tape import data_utils  # noqa: E402
import tensorflow as tf  # noqa: E402
import numpy as np  # noqa: E402


def pythonify(tensor):
    array = tensor.numpy()
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, bytes):
        return array
    elif isinstance(array, (int, np.int32, np.int64)):
        return int(array)
    else:
        raise ValueError(array)


def convert(flist: Union[Path, List[Path]], outfile: Path, deserialization_func):
    files = str(flist) if not isinstance(flist, list) else [str(path) for path in flist]
    data = tf.data.TFRecordDataset(files).map(deserialization_func)
    vocab = {v: k for k, v in data_utils.PFAM_VOCAB.items()}

    env = lmdb.open(str(outfile), map_size=50e9)
    with env.begin(write=True) as txn:
        num_examples = 0
        for i, example in enumerate(tqdm(data)):
            item = {name: pythonify(tensor) for name, tensor in example.items()}
            item['primary'] = ''.join(vocab[index] for index in item['primary'])
            id_ = str(i).encode()
            txn.put(id_, pkl.dumps(item))
            num_examples += 1
        txn.put(b'num_examples', pkl.dumps(num_examples))


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow info logging
tf.enable_eager_execution()

data_dir = Path('/home/rmrao/projects/tape/data/')
out_dir = Path('/home/rmrao/projects/tape-pytorch/data')

file_lists: List[Union[Path, List[Path]]] = []
file_lists.append(list((data_dir / 'pfam').glob('pfam31_train*.tfrecord')))
file_lists.append(list((data_dir / 'pfam').glob('pfam31_valid*.tfrecord')))
file_lists.append(data_dir / 'pfam' / 'pfam31_holdout.tfrecord')
file_lists.append(list((data_dir / 'proteinnet').glob('contact_map_train*.tfrecord')))
file_lists.append(data_dir / 'proteinnet' / 'contact_map_valid.tfrecord')
file_lists.append(data_dir / 'proteinnet' / 'contact_map_test.tfrecord')
file_lists += list((data_dir / 'fluorescence').glob('*.tfrecord'))
file_lists += list((data_dir / 'stability').glob('*.tfrecord'))
file_lists += list((data_dir / 'remote_homology').glob('*.tfrecord'))
file_lists += list((data_dir / 'secondary_structure').glob('*.tfrecord'))

deserialize_funcs = {
    'pfam': data_utils.deserialize_pfam_sequence,
    'proteinnet': data_utils.deserialize_proteinnet_sequence,
    'fluorescence': data_utils.deserialize_gfp_sequence,
    'stability': data_utils.deserialize_stability_sequence,
    'remote_homology': data_utils.deserialize_remote_homology_sequence,
    'secondary_structure': data_utils.deserialize_secondary_structure}

# outfile = 'data/proteinnet/proteinnet_test.lmdb'
flist_names = ['pfam_train.lmdb', 'pfam_valid.lmdb', 'proteinnet_train.lmdb']

for flist in file_lists:
    if isinstance(flist, list):
        name = flist_names.pop(0)
        task_name = flist[0].relative_to(data_dir).parts[0]
        deserialization_func = deserialize_funcs[task_name]
    else:
        name = flist.with_suffix('.lmdb').name
        task_name = flist.relative_to(data_dir).parts[0]
        deserialization_func = deserialize_funcs[task_name]

    name = name.replace('pfam31', 'pfam')
    name = name.replace('contact_map', 'proteinnet')

    outfile = out_dir / task_name / name
    if outfile.exists():
        continue

    print("Converting", name)
    convert(flist, outfile, deserialization_func)
