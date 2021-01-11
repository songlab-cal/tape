from abc import abstractmethod, abstractstaticmethod, abstractclassmethod
from argparse import ArgumentParser, Namespace
import logging

import torch.distributed as dist
import pytorch_lightning as pl

from .optimization import BaseOptimizationMixin

logger = logging.getLogger("base")


class BaseTrainingModule(pl.LightningModule, BaseOptimizationMixin):
    r""" BaseTrainingModule subclasses the pl.LightningModule and implements scaffolding for
    training, optimization, etc.

    To implement a model in this framework, subclass BaseTrainingModule and implement
    the `__init__`, `forward` and `add_argparse_args`, and `from_argparse_args` methods.
    """

    @abstractmethod
    def forward(self, **kwargs):
        return NotImplemented

    @abstractclassmethod
    def from_argparse_args(cls, hparams: Namespace) -> "BaseTrainingModule":
        pass

    @abstractstaticmethod
    def add_argparse_args(parser: ArgumentParser) -> ArgumentParser:
        pass

    def _run_forward(self, batch):
        if isinstance(batch, dict):
            loss, *outputs = self.forward(**batch)
        elif isinstance(batch, (tuple, list)):
            loss, *outputs = self.forward(*batch)
        else:
            loss, *outputs = self.forward(batch)
        return loss, outputs

    def training_step(self, batch, batch_nb):
        loss, outputs = self._run_forward(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.compute_and_log_metrics(batch, loss, *outputs, mode="train")
        return loss

    def validation_step(self, batch, batch_nb):
        loss, outputs = self._run_forward(batch)
        self.log("valid/loss", loss)
        self.compute_and_log_metrics(batch, loss, *outputs, mode="valid")
        return loss

    @property
    def is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized()
