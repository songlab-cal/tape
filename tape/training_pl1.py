from argparse import ArgumentParser
import os
import sys
from typing import Optional, List
import logging
import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger

from . import registry
from pytorch_lightning.loggers import WandbLogger
from .visualization import LoggingProgressBar
from .config import create_parser
from .base import BaseTrainingModule


def sanitize_hparams(hparams: argparse.Namespace) -> argparse.Namespace:
    for name, val in vars(hparams).items():
        if not isinstance(val, (int, float, str, bool, torch.Tensor)):
            setattr(hparams, name, str(val))
    return hparams


def setup_logging(hparams) -> None:

    level = hparams.log_level
    if isinstance(level, str):
        level = level.upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    if root_logger.hasHandlers():
        console_handler = root_logger.handlers[0]
    else:
        console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def get_gpus(hparams: argparse.Namespace) -> List[int]:
    if hparams.gpu_list is not None:
        gpus = [int(gpu) for gpu in hparams.gpu_list]
    else:
        gpus = list(range(hparams.gpus))

    hparams.gpus = len(gpus)
    return gpus


def maybe_unset_distributed(hparams: argparse.Namespace) -> Optional[str]:
    if hparams.gpus > 1:
        return hparams.distributed_backend
    else:
        hparams.distributed_backend = ""
        return None


def is_debug_run(hparams: argparse.Namespace) -> bool:
    parser = create_parser()
    debug_params = [
        "overfit_pct",
        "fast_dev_run",
        "train_percent_check",
        # "val_percent_check",
        # "test_percent_check",
        "print_nan_grads",
    ]
    return any(
        getattr(hparams, param) != parser.get_default(param) for param in debug_params
    )


def train(hparams: argparse.Namespace):
    """Train the model. Log training results to W&B. Takes in the output of
    create_parser().parse_args().
    """
    setup_logging(hparams)
    task = tasks.from_argparse_args(hparams)
    model = models.from_argparse_args(hparams)
    trainer = Trainer.from_argparse_args(hparams)
    trainer.fit(model, task.train_dataloader(), task.valid_dataloader())


def main():
    parser = ArgumentParser(desc="Train a Sequence-to-Sequence model on protein data")
    parser.add_argument("task", type=str, help="Which task to train.")
    parser.add_argument("model", type=str, help="Which model to use for training.")
    known_args = parser.parse_known_args()
    task = tasks.get(known_args.task)
    model = task.get(model)
    Trainer.add_argparse_args(parser)
    model.add_argparse_args(parser)
    task.add_argparse_args(parser)
    hparams = parser.parse_args()
    train(hparams)


if __name__ == "__main__":
    main()
