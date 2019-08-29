import tensorflow as tf
import numpy as np
from tape import data_utils
import lmdb
import os
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow info logging
tf.enable_eager_execution()


data_dir = Path('/home/rmrao/projects/tape/data/')

files = str(data_dir / 'proteinnet' / 'contact_map_valid.tfrecord')

outfile = 'data/proteinnet/proteinnet_test.lmdb'
data = tf.data.TFRecordDataset(files).map(data_utils.deserialize_proteinnet_sequence)


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


env = lmdb.open(str(outfile), map_size=50e9)
id_list = []
with env.begin(write=True) as txn:
    for i, example in enumerate(tqdm(data)):
        item = {name: pythonify(tensor) for name, tensor in example.items()}
        id_ = str(i).encode()
        txn.put(id_, pkl.dumps(item))
        id_list.append(id_)
    txn.put('keys'.encode(), pkl.dumps(id_list))
