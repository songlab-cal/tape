import lmdb
from tqdm import tqdm
import pickle as pkl
from pathlib import Path
from tape.data_utils import PFAM_VOCAB
import numpy as np


vocab = {v: k for k, v in PFAM_VOCAB.items()}


files = Path('data').rglob('*.lmdb')

for lmdbfile in files:
    print(lmdbfile)
    env = lmdb.open(str(lmdbfile), map_size=50e9)
    with env.begin(write=True) as txn:
        keys = pkl.loads(txn.get(b'keys'))
        for key in tqdm(keys):
            data = txn.get(key)
            item = pkl.loads(txn.get(key))
            if isinstance(item['primary'], np.ndarray):
                item['primary'] = ''.join(vocab[index] for index in item['primary'])
                txn.replace(key, pkl.dumps(item))
