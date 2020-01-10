import typing
import random
from pathlib import Path
import logging
from time import strftime, gmtime
from datetime import datetime
import os
import argparse
import contextlib
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

logger = logging.getLogger(__name__)
FloatOrTensor = typing.Union[float, torch.Tensor]


def int_or_str(arg: str) -> typing.Union[int, str]:
    try:
        return int(arg)
    except ValueError:
        return arg


def check_is_file(file_path: str) -> str:
    if file_path is None or os.path.isfile(file_path):
        return file_path
    else:
        raise argparse.ArgumentTypeError(f"File path: {file_path} is not a valid file")


def check_is_dir(dir_path: str) -> str:
    if dir_path is None or os.path.isdir(dir_path):
        return dir_path
    else:
        raise argparse.ArgumentTypeError(f"Directory path: {dir_path} is not a valid directory")


def path_to_datetime(path: Path) -> datetime:
    name = path.name
    datetime_string = name.split('_')[0]
    try:
        year, month, day, hour, minute, second = datetime_string.split('-')
    except ValueError:
        try:
            # Deprecated datetime strings
            year, month, day, time_str = datetime_string.split('-')
            hour, minute, second = time_str.split(':')
        except ValueError:
            return datetime(1, 1, 1)

    pathdatetime = datetime(
        int(year), int(month), int(day), int(hour), int(minute), int(second))
    return pathdatetime


def get_expname(exp_name: typing.Optional[str],
                task: typing.Optional[str] = None,
                model_type: typing.Optional[str] = None) -> str:
    if exp_name is None:
        time_stamp = strftime("%y-%m-%d-%H-%M-%S", gmtime())
        exp_name = f"{task}_{model_type}_{time_stamp}_{random.randint(0, int(1e6)):0>6d}"
    return exp_name


def set_random_seeds(seed: int, n_gpu: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)  # type: ignore


def get_effective_num_gpus(local_rank: int, n_gpu: int) -> int:
    if local_rank == -1:
        num_gpus = n_gpu
    else:
        num_gpus = dist.get_world_size()
    return num_gpus


def get_effective_batch_size(batch_size: int,
                             local_rank: int,
                             n_gpu: int,
                             gradient_accumulation_steps: int = 1) -> int:
    eff_batch_size = float(batch_size)
    eff_batch_size /= gradient_accumulation_steps
    eff_batch_size /= get_effective_num_gpus(local_rank, n_gpu)
    return int(eff_batch_size)


def get_num_train_optimization_steps(dataset: Dataset,
                                     batch_size: int,
                                     num_train_epochs: int) -> int:
    return int(len(dataset) / batch_size * num_train_epochs)


class MetricsAccumulator:

    def __init__(self, smoothing: float = 0.95):
        self._loss_tmp = 0.
        self._smoothloss: typing.Optional[float] = None
        self._totalloss = 0.
        self._metricstmp: typing.Dict[str, float] = defaultdict(lambda: 0.0)
        self._smoothmetrics: typing.Dict[str, float] = {}
        self._totalmetrics: typing.Dict[str, float] = defaultdict(lambda: 0.0)

        self._nacc_steps = 0
        self._nupdates = 0
        self._smoothing = smoothing

    def update(self,
               loss: FloatOrTensor,
               metrics: typing.Dict[str, FloatOrTensor],
               step: bool = True) -> None:
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        self._loss_tmp += loss
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self._metricstmp[name] += value
        self._nacc_steps += 1

        if step:
            self.step()

    def step(self) -> typing.Dict[str, float]:
        loss_tmp = self._loss_tmp / self._nacc_steps
        metricstmp = {name: value / self._nacc_steps
                      for name, value in self._metricstmp.items()}

        if self._smoothloss is None:
            self._smoothloss = loss_tmp
        else:
            self._smoothloss *= self._smoothing
            self._smoothloss += (1 - self._smoothing) * loss_tmp
        self._totalloss += loss_tmp

        for name, value in metricstmp.items():
            if name in self._smoothmetrics:
                currvalue = self._smoothmetrics[name]
                newvalue = currvalue * self._smoothing + value * (1 - self._smoothing)
            else:
                newvalue = value

            self._smoothmetrics[name] = newvalue
            self._totalmetrics[name] += value

        self._nupdates += 1

        self._nacc_steps = 0
        self._loss_tmp = 0
        self._metricstmp = defaultdict(lambda: 0.0)

        metricstmp['loss'] = loss_tmp
        return metricstmp

    def loss(self) -> float:
        if self._smoothloss is None:
            raise RuntimeError("Trying to get the loss without any updates")
        return self._smoothloss

    def metrics(self) -> typing.Dict[str, float]:
        if self._nupdates == 0:
            raise RuntimeError("Trying to get metrics without any updates")
        return dict(self._smoothmetrics)

    def final_loss(self) -> float:
        return self._totalloss / self._nupdates

    def final_metrics(self) -> typing.Dict[str, float]:
        return {name: value / self._nupdates
                for name, value in self._totalmetrics.items()}


class wrap_cuda_oom_error(contextlib.ContextDecorator):
    """A context manager that wraps the Cuda OOM message so that you get some more helpful
    context as to what you can/should change. Can also be used as a decorator.

    Examples:
        1) As a context manager:

            with wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation):
                loss = model.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad

        2) As a decorator:

            @wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation)
            def run_train_epoch(args):
                ...
                <code to run training epoch>
                ...
    """

    def __init__(self,
                 local_rank: int,
                 batch_size: int,
                 n_gpu: int = 1,
                 gradient_accumulation_steps: typing.Optional[int] = None):
        self._local_rank = local_rank
        self._batch_size = batch_size
        self._n_gpu = n_gpu
        self._gradient_accumulation_steps = gradient_accumulation_steps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        exc_args = exc_value.args if exc_value is not None else None
        if exc_args and 'CUDA out of memory' in exc_args[0]:
            eff_ngpu = get_effective_num_gpus(self._local_rank, self._n_gpu)
            if self._gradient_accumulation_steps is not None:
                eff_batch_size = get_effective_batch_size(
                    self._batch_size, self._local_rank, self._n_gpu,
                    self._gradient_accumulation_steps)
                message = (f"CUDA out of memory. Reduce batch size or increase "
                           f"gradient_accumulation_steps to divide each batch over more "
                           f"forward passes.\n\n"
                           f"\tHyperparameters:\n"
                           f"\t\tbatch_size per backward-pass: {self._batch_size}\n"
                           f"\t\tgradient_accumulation_steps: "
                           f"{self._gradient_accumulation_steps}\n"
                           f"\t\tn_gpu: {eff_ngpu}\n"
                           f"\t\tbatch_size per (gpu * forward-pass): "
                           f"{eff_batch_size}")
            else:
                eff_batch_size = get_effective_batch_size(
                    self._batch_size, self._local_rank, self._n_gpu)
                message = (f"CUDA out of memory. Reduce batch size to fit each "
                           f"iteration in memory.\n\n"
                           f"\tHyperparameters:\n"
                           f"\t\tbatch_size per forward-pass: {self._batch_size}\n"
                           f"\t\tn_gpu: {eff_ngpu}\n"
                           f"\t\tbatch_size per (gpu * forward-pass): "
                           f"{eff_batch_size}")
            raise RuntimeError(message)
        return False


def write_lmdb(filename: str, iterable: typing.Iterable, map_size: int = 2 ** 20):
    """Utility for writing a dataset to an LMDB file.

    Args:
        filename (str): Output filename to write to
        iterable (Iterable): An iterable dataset to write to. Entries must be pickleable.
        map_size (int, optional): Maximum allowable size of database in bytes. Required by LMDB.
            You will likely have to increase this. Default: 1MB.
    """
    import lmdb
    import pickle as pkl
    env = lmdb.open(filename, map_size=map_size)

    with env.begin(write=True) as txn:
        for i, entry in enumerate(iterable):
            txn.put(str(i).encode(), pkl.dumps(entry))
        txn.put(b'num_examples', pkl.dumps(i + 1))
    env.close()


class IncrementalNPZ(object):
    # Modified npz that allows incremental saving, from https://stackoverflow.com/questions/22712292/how-to-use-numpy-savez-in-a-loop-for-save-more-than-one-array  # noqa: E501
    def __init__(self, file):
        import tempfile
        import zipfile
        import os

        if isinstance(file, str):
            if not file.endswith('.npz'):
                file = file + '.npz'

        compression = zipfile.ZIP_STORED

        zipfile = self.zipfile_factory(file, mode="w", compression=compression)

        # Stage arrays in a temporary file on disk, before writing to zip.
        fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
        os.close(fd)

        self.tmpfile = tmpfile
        self.zip = zipfile
        self._i = 0

    def zipfile_factory(self, *args, **kwargs):
        import zipfile
        import sys
        if sys.version_info >= (2, 5):
            kwargs['allowZip64'] = True
        return zipfile.ZipFile(*args, **kwargs)

    def savez(self, *args, **kwds):
        import os
        import numpy.lib.format as fmt

        namedict = kwds
        for val in args:
            key = 'arr_%d' % self._i
            if key in namedict.keys():
                raise ValueError("Cannot use un-named variables and keyword %s" % key)
            namedict[key] = val
            self._i += 1

        try:
            for key, val in namedict.items():
                fname = key + '.npy'
                fid = open(self.tmpfile, 'wb')
                with open(self.tmpfile, 'wb') as fid:
                    fmt.write_array(fid, np.asanyarray(val), allow_pickle=True)
                self.zip.write(self.tmpfile, arcname=fname)
        finally:
            os.remove(self.tmpfile)

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
