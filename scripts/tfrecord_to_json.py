import sys
import tensorflow as tf
import numpy as np
import tape.data_utils
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # disable tensorflow info logging
tf.enable_eager_execution()

filename = sys.argv[1]
funcname = sys.argv[2]
outfile = filename.rsplit('.', 1)[0] + '.json'

func = getattr(tape.data_utils, funcname)
data = tf.data.TFRecordDataset(filename).map(func)


def pythonify(tensor):
    array = tensor.numpy()
    if isinstance(array, np.ndarray):
        return array.tolist()
    elif isinstance(array, bytes):
        return array.decode()
    elif isinstance(array, (int, np.int32, np.int64)):
        return int(array)
    else:
        raise ValueError(array)


jsondata = [{name: pythonify(tensor) for name, tensor in ex.items()} for ex in data]

with open(outfile, 'w') as f:
    json.dump(jsondata, f)
