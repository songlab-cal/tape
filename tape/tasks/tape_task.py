from typing import Callable, Optional
import logging
from pathlib import Path
import tempfile
import tarfile
from argparse import ArgumentParser
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from .. import lr_schedulers
from ..utils import http_get, PathLike

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parents[2] / "data"


class TAPEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_url: str,
        task_name: str,
        data_dir: PathLike = DEFAULT_DATA_DIR,
        batch_size: int = 64,
        num_workers: int = 3,
        tokenizer: str = "iupac",
    ):
        super().__init__()
        self._data_url = data_url
        self._task_name = task_name
        self._data_dir = Path(data_dir)
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--data_dir", default=DEFAULT_DATA_DIR, help="Data directory"
        )
        parser.add_argument("--tokenizer", default="tokenizer", type=str)
        parser.add_argument(
            "--batch_size", default=64, type=int, help="Batch size during training."
        )
        parser.add_argument(
            "--num_workers", default=3, type=int, help="Num dataloading workers."
        )
        return parser

    def prepare_data(self):
        if (self.data_dir / self.task_name).exists():
            logger.info("Data found, not downloading.")
        else:
            with tempfile.NamedTemporaryFile() as temp_file:
                logger.info(f"Downloading data from {self.data_url}")
                http_get(self.data_url, temp_file, "Downloading Data")
                temp_file.flush()
                logger.info(f"Extracting data to {self.data_dir}")
                with tarfile.open(temp_file.name, "r|gz") as f:
                    f.extractall(self.data_dir)

    def make_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
        )

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def data_url(self) -> str:
        return self._data_url

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def tokenizer(self) -> str:
        return self._tokenizer

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_workers(self) -> int:
        return self._num_workers


class TAPEPredictorBase(pl.LightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        extract_features: Callable[
            [nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
        ],
        embedding_dim: int,
        freeze_base: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "constant",
        warmup_steps: int = 0,
        max_steps: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters(
            "freeze_base",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "lr_scheduler",
            "warmup_steps",
            "max_steps",
        )
        self.save_hyperparameters({"base_model": base_model.__class__.__name__})
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        if freeze_base:
            base_model.requires_grad_(False)

        self.base_model = base_model
        self.extract_features = partial(extract_features, self.base_model)

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--freeze-base",
            action="store_true",
            default=True,
            help="Freeze base model weights.",
        )
        parser.add_argument(
            "--no-freeze-base",
            action="store_false",
            dest="freeze_base",
            help="Freeze base model weights.",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            help="Learning rate for training.",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-4,
            help="Weight Decay Coefficient.",
        )
        parser.add_argument(
            "--optimizer",
            choices=["adam", "lamb"],
            default="adam",
            help="Which optimizer to use.",
        )
        parser.add_argument(
            "--lr_scheduler",
            choices=lr_schedulers.LR_SCHEDULERS.keys(),
            default="warmup_constant",
            help="Learning rate scheduler to use.",
        )
        parser.add_argument(
            "--warmup_steps",
            type=int,
            default=0,
            help="How many warmup steps to use when using a warmup schedule.",
        )
        return parser

    def forward(self, src_tokens, src_lengths):
        # B x L x D
        features = self.extract_features(src_tokens, src_lengths)
        return features

    def configure_optimizers(self):
        no_decay = ["norm", "LayerNorm"]

        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.optimizer == "adam":
            optimizer_type = torch.optim.AdamW
        elif self.optimizer == "lamb":
            try:
                from apex.optimizers import FusedLAMB
            except ImportError:
                raise ImportError("Apex must be installed to use FusedLAMB optimizer.")
            optimizer_type = FusedLAMB
        optimizer = optimizer_type(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = lr_schedulers.get(self.lr_scheduler)(
            optimizer, self.warmup_steps, self.max_steps
        )

        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]
