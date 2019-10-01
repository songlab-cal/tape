from typing import Union, Optional, Tuple, TypeVar
import random
import sys
from pathlib import Path
import logging
from time import strftime, gmtime
from datetime import datetime
import os
import argparse

import numpy as np
import torch
from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


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


def setup_logging(local_rank: int, save_path: Optional[Path] = None) -> None:
    log_level = logging.INFO if local_rank in (-1, 0) else logging.WARNING
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if save_path is not None:
        file_handler = logging.FileHandler(save_path / 'log')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def path_to_datetime(path: Path) -> datetime:
    name = path.name
    datetime_string = name.split('_')[0]
    try:
        year, month, day, hour, minute, second = datetime_string.split('-')
    except ValueError:
        # Deprecated datetime strings
        year, month, day, time_str = datetime_string.split('-')
        hour, minute, second = time_str.split(':')

    pathdatetime = datetime(
        int(year), int(month), int(day), int(hour), int(minute), int(second))
    return pathdatetime


def get_expname(exp_name: Optional[str]) -> str:
    if exp_name is None:
        time_stamp = strftime("%y-%m-%d-%H-%M-%S", gmtime())
        exp_name = time_stamp + "_{:0>6d}".format(random.randint(0, int(1e6)))
    return exp_name


def get_savepath_and_expname(output_dir: str,
                             exp_name: Optional[str],
                             is_master: bool = True) -> Tuple[Path, str]:
    if is_master:
        exp_name = get_expname(exp_name)
        save_path = Path(output_dir) / exp_name
        save_path.mkdir(parents=True, exist_ok=False)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        save_files = Path(output_dir).iterdir()
        save_path = max(save_files, key=path_to_datetime)
        exp_name = save_path.name

    return save_path, exp_name


def set_random_seeds(seed: int, n_gpu: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class TBLogger:

    def __init__(self, log_dir: Union[str, Path], exp_name: str, local_rank: int):
        is_master = local_rank in (-1, 0)
        if is_master:
            log_dir = Path(log_dir) / exp_name
            logger.info(f"tensorboard file at: {log_dir}")
            self.logger = SummaryWriter(log_dir=str(log_dir))
        self._is_master = is_master

    def line_plot(self, step, val, split, key, xlabel="None") -> None:
        if self._is_master:
            self.logger.add_scalar(split + "/" + key, val, step)


class MetricsAccumulator:

    def __init__(self, smoothing: float = 0.95):
        self._currloss: Optional[float] = None
        self._totalloss = 0.
        self._nupdates = 0
        self._smoothing = smoothing

    def update(self, loss: float):
        if self._currloss is None:
            self._currloss = loss
        else:
            self._currloss = (self._smoothing) * self._currloss + (1 - self._smoothing) * loss

        self._totalloss += loss
        self._nupdates += 1

    def loss(self) -> float:
        if self._currloss is None:
            raise RuntimeError("Trying to get the loss without any updates")
        return self._currloss

    def final_loss(self) -> float:
        return self._totalloss / self._nupdates
