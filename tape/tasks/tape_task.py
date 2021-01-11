from abc import ABC, abstractproperty, abstractmethod, abstractclassmethod
from typing import Callable, Optional, Union, Dict, List, Type
import logging
from pathlib import Path
import tempfile
import tarfile
from argparse import ArgumentParser, Namespace
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from .. import lr_schedulers
from ..tokenizers import TAPETokenizer, FairseqTokenizer
from ..datasets import LMDBDataset
from ..utils import http_get, PathLike, TensorDict

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parents[2] / "data"


class TAPEDataset(Dataset, ABC):
    def __init__(
        self,
        data_path: PathLike,
        split: str,
        tokenizer: Union[str, TAPETokenizer] = "iupac",
        use_msa: bool = False,
        max_tokens_per_msa: int = 2 ** 14,
    ):
        super().__init__()
        if split not in self.splits:
            raise ValueError(
                f"Unrecognized split: {split}. " f"Must be one of {self.splits}"
            )
        if isinstance(tokenizer, str):
            if Path(tokenizer).exists() and tokenizer.endswith("dict.txt"):
                tokenizer = FairseqTokenizer(tokenizer)
            else:
                tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f"{self.task_name}/{self.task_name}_{split}.lmdb"
        self.data = LMDBDataset(data_path / data_file)
        self.data_path = data_path
        self.use_msa = use_msa
        self.max_tokens_per_msa = max_tokens_per_msa

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError

    @abstractmethod
    def collate_fn(
        self, batch: List[TensorDict]
    ) -> Dict[str, Union[torch.Tensor, TensorDict]]:
        raise NotImplementedError

    @abstractproperty
    def task_name(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def splits(self) -> List[str]:
        raise NotImplementedError


class TAPEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: PathLike = DEFAULT_DATA_DIR,
        batch_size: int = 64,
        num_workers: int = 3,
        tokenizer: str = "iupac",
        use_msa: bool = False,
        max_tokens_per_msa: int = 2 ** 14,
    ):
        super().__init__()
        self._data_dir = Path(data_dir)
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._use_msa = use_msa
        self._max_tokens_per_msa = max_tokens_per_msa

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--data_dir", default=DEFAULT_DATA_DIR, help="Data directory"
        )
        parser.add_argument("--tokenizer", default="iupac", type=str)
        parser.add_argument(
            "--batch_size", default=64, type=int, help="Batch size during training."
        )
        parser.add_argument(
            "--num_workers", default=3, type=int, help="Num dataloading workers."
        )
        parser.add_argument(
            "--use_msa",
            action="store_true",
            help="Whether to pass MSAs to the input model.",
        )
        parser.add_argument(
            "--max_tokens_per_msa",
            type=int,
            default=2 ** 14,
            help="Max tokens to use in the MSA if loading MSAs.",
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

    def make_dataloader(
        self, split: str, shuffle: bool = False
    ) -> DataLoader:
        dataset = self.dataset_type(
            self.data_dir, split, self.tokenizer, self.use_msa, self.max_tokens_per_msa
        )
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

    @abstractproperty
    def data_url(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def task_name(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def dataset_type(self) -> Type[TAPEDataset]:
        raise NotImplementedError

    @property
    def tokenizer(self) -> str:
        return self._tokenizer

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_workers(self) -> int:
        return self._num_workers

    @property
    def use_msa(self) -> bool:
        return self._use_msa

    @property
    def max_tokens_per_msa(self) -> int:
        return self._max_tokens_per_msa

    def train_dataloader(self):
        return self.make_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self.make_dataloader("valid", shuffle=True)

    def test_dataloader(self):
        return self.make_dataloader("test", shuffle=True)


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

    @abstractclassmethod
    def from_argparse_args(
        cls,
        args: Namespace,
        base_model: nn.Module,
        extract_features: Callable[
            [nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
        ],
        embedding_dim: int,
    ):
        raise NotImplementedError

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


class TAPETask:
    def __init__(
        self,
        task_name: str,
        task_data_type: Type[TAPEDataModule],
        task_model_type: Type[TAPEPredictorBase],
    ):
        self.task_name = task_name
        self.task_data_type = task_data_type
        self.task_model_type = task_model_type

    def __str__(self):
        return f"TAPETask: {self.task_name}"

    def add_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser = self.task_data_type.add_args(parser)
        parser = self.task_model_type.add_args(parser)
        return parser

    def build_data(self, args: Namespace) -> TAPEDataModule:
        return self.task_data_type(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tokenizer=args.tokenizer,
            use_msa=args.use_msa,
            max_tokens_per_msa=args.max_tokens_per_msa,
        )

    def build_model(
        self,
        args: Namespace,
        base_model: nn.Module,
        extract_features,
        embedding_dim: int,
    ) -> TAPEPredictorBase:
        return self.task_model_type.from_argparse_args(  # type: ignore
            args=args,
            base_model=base_model,
            extract_features=extract_features,
            embedding_dim=embedding_dim,
        )
