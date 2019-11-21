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
import torch.nn as nn
from torch.utils.data import Dataset
import torch.distributed as dist

try:
    from apex import amp
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False


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


def resume_from_checkpoint(from_pretrained: str,
                           optimizer: torch.optim.Optimizer,  # type: ignore
                           scheduler: torch.optim.lr_scheduler.LambdaLR,
                           device: torch.device,
                           fp16: bool) -> int:
    checkpoint = torch.load(
        os.path.join(from_pretrained, 'checkpoint.bin'), map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if fp16:
        assert APEX_FOUND
        optimizer._lazy_init_maybe_master_weights()
        optimizer._amp_stash.lazy_init_called = True
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param, saved in zip(amp.master_params(optimizer), checkpoint['master params']):
            param.data.copy_(saved.data)
        amp.load_state_dict(checkpoint['amp'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


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
                 gradient_accumulation_steps: int = 1):
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
            eff_batch_size = get_effective_batch_size(
                self._batch_size, self._local_rank, self._n_gpu,
                self._gradient_accumulation_steps)
            message = (f"CUDA out of memory. Increase gradient_accumulation_steps to "
                       f"divide each batch over more forward passes.\n\n"
                       f"\tHyperparameters:\n"
                       f"\t\tbatch_size per backward-pass: {self._batch_size}\n"
                       f"\t\tgradient_accumulation_steps: "
                       f"{self._gradient_accumulation_steps}\n"
                       f"\t\tn_gpu: {eff_ngpu}\n"
                       f"\t\tbatch_size per (gpu * forward-pass): "
                       f"{eff_batch_size}")
            raise RuntimeError(message)
        return False


def save_state(save_directory: typing.Union[str, Path],
               model: nn.Module,
               optimizer: torch.optim.Optimizer,  # type: ignore
               scheduler: torch.optim.lr_scheduler.LambdaLR,
               epoch: int) -> None:
    save_directory = Path(save_directory)
    if not save_directory.exists():
        save_directory.mkdir()
    else:
        assert save_directory.is_dir(), "Save path should be a directory"
    model_to_save = getattr(model, 'module', model)
    model_to_save.save_pretrained(save_directory)
    optimizer_state: typing.Dict[str, typing.Any] = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch}
    if APEX_FOUND:
        optimizer_state['master params'] = list(amp.master_params(optimizer))
        try:
            optimizer_state['amp'] = amp.state_dict()
        except AttributeError:
            pass
    torch.save(optimizer_state, save_directory / 'checkpoint.bin')
