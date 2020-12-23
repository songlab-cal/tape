from typing import Union, Dict, List
import logging
from pathlib import Path
import tempfile
import tarfile
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from tape.datasets import LMDBDataset
from tape.tokenizers import TAPETokenizer
from . import lr_schedulers
from .utils import http_get, pad_sequences

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
DEFAULT_DATA_DIR = Path(__file__).parents[1] / "data"
TensorDict = Dict[str, torch.Tensor]


class SecondaryStructureDataset(Dataset):
    def __init__(
        self,
        data_path: PathLike,
        split: str,
        tokenizer: Union[str, TAPETokenizer] = "iupac",
        in_memory: bool = False,
    ):
        super().__init__()

        if split not in ("train", "valid", "casp12", "ts115", "cb513"):
            raise ValueError(
                f"Unrecognized split: {split}. Must be one of "
                f"['train', 'valid', 'casp12', "
                f"'ts115', 'cb513']"
            )
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f"secondary_structure/secondary_structure_{split}.lmdb"
        self.data = LMDBDataset(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> TensorDict:
        item = self.data[index]
        src_tokens = torch.from_numpy(self.tokenizer.encode(item["primary"])).long()
        ss3 = torch.from_numpy(item["ss3"]).long()
        ss8 = torch.from_numpy(item["ss8"]).long()
        rsa = torch.from_numpy(item["rsa"]).float()

        phi_rad = torch.from_numpy(np.deg2rad(item["phi"])).float()
        phi = torch.stack([phi_rad.sin(), phi_rad.cos()], 1)
        psi_rad = torch.from_numpy(np.deg2rad(item["psi"])).float()
        psi = torch.stack([psi_rad.sin(), psi_rad.cos()], 1)

        disorder = torch.from_numpy(item["disorder"]).float()
        interface = torch.from_numpy(item["interface"]).float()
        valid_mask = torch.from_numpy(item["valid_mask"]).bool()

        return {
            "src_tokens": src_tokens,
            "Q8": ss8,
            "Q3": ss3,
            "RSA": rsa,
            "Phi": phi,
            "Psi": psi,
            "Disorder": disorder,
            "Interface": interface,
            "valid_mask": valid_mask,
        }

    def collate_fn(
        self, batch: List[TensorDict]
    ) -> Dict[str, Union[torch.Tensor, TensorDict]]:
        src_tokens = pad_sequences([el["src_tokens"] for el in batch])
        src_lengths = torch.tensor([el["src_tokens"] for el in batch], dtype=torch.long)

        result = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
            "Q8": pad_sequences([el["Q8"] for el in batch], constant_value=-1),
            "Q3": pad_sequences([el["Q3"] for el in batch], constant_value=-1),
            "RSA": pad_sequences([el["RSA"] for el in batch]),
            "Phi": pad_sequences([el["Phi"] for el in batch]),
            "Psi": pad_sequences([el["Psi"] for el in batch]),
            "Disorder": pad_sequences([el["Disorder"] for el in batch]),
            "Interface": pad_sequences([el["Interface"] for el in batch]),
            "valid_mask": pad_sequences([el["valid_mask"] for el in batch]),
        }
        return result


class SecondaryStructureDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: PathLike = DEFAULT_DATA_DIR,
        batch_size: int = 64,
        num_workers: int = 10,
    ):
        super().__init__()
        self._data_dir = Path(data_dir)
        self._data_url = "http://s3.amazonaws.com/proteindata/data_pytorch/secondary_structure.tar.gz"  # noqa: E501

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--data_dir", default=DEFAULT_DATA_DIR, help="Data directory"
        )
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
            "--batch_size", default=64, type=int, help="Batch size during training."
        )
        parser.add_argument(
            "--num_workers", default=10, type=int, help="Num dataloading workers."
        )
        return parser

    def prepare_data(self):
        if (self.data_dir / "secondary_structure").exists():
            logger.info("Data found, not downloading.")
        else:
            with tempfile.NamedTemporaryFile() as temp_file:
                logger.info(f"Downloading data from {self.data_url}")
                http_get(self.data_url, temp_file, "Downloading Data")
                temp_file.flush()
                logger.info(f"Extracting data to {self.data_dir}")
                with tarfile.open(temp_file.name, "r|gz") as f:
                    f.extractall(self.data_dir)

    def train_dataloader(self):
        DataLoader(
            SecondaryStructureDataset(self.data_dir, "train", self.tokenizer),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        DataLoader(
            SecondaryStructureDataset(self.data_dir, "valid", self.tokenizer),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        for test_set in ["cb513", "ts115", "casp12"]:
            yield DataLoader(
                SecondaryStructureDataset(self.data_dir, test_set, self.tokenizer),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def data_url(self) -> str:
        return self._data_url


class SecondaryStructurePrediction(pl.LightningModule):
    def __init__(
        self,
        base_model: nn.Module,
        embedding_dim: int,
        conv_dropout: float = 0.1,
        lstm_dropout: float = 0.1,
        freeze_base: bool = False,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "constant",
        warmup_steps: int = 0,
        max_steps: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        output_names = ["Q8", "Q3", "RSA", "Phi", "Psi", "Disorder", "Interface"]
        output_sizes = [8, 3, 1, 2, 2, 1, 1]
        if freeze_base:
            base_model.requires_grad_(False)

        self.base_model = base_model
        self.conv1 = nn.Sequential(
            nn.Dropout(conv_dropout),
            nn.Conv1d(embedding_dim, 32, kernel_size=129, padding=64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(conv_dropout),
            nn.Conv1d(embedding_dim, 32, kernel_size=257, padding=128),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            embedding_dim + 64,
            1024,
            num_layers=2,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.outproj = nn.Linear(2048, sum(output_sizes))
        self.output_names = output_names
        self.output_sizes = output_sizes

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--conv_dropout",
            type=float,
            default=0.1,
            help="Dropout on conv layers",
        )
        parser.add_argument(
            "--lstm_dropout",
            type=float,
            default=0.1,
            help="Dropout on conv layers",
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
        features = self.base_model(src_tokens, src_lengths)
        # B x L x D -> B x D x L
        features = features.transpose(1, 2)
        conv1_out = self.conv1(features)
        conv2_out = self.conv2(features)
        features = torch.cat([features, conv1_out, conv2_out], 1)
        # B x D x L -> L x B x D
        features = features.permute(2, 0, 1)
        lstm_input = rnn.pack_padded_sequence(features, src_lengths)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_output, _ = rnn.pad_packed_sequence(lstm_output)

        output = self.outproj(lstm_output)
        output = output.split(self.output_sizes, dim=-1)
        return dict(zip(self.output_names, output))

    def _masked_loss(self, unreduced_loss, *masks):
        dtype = unreduced_loss.dtype
        mask = masks[0].to(dtype)
        for addmask in masks[1:]:
            mask *= addmask.to(dtype)

        return (unreduced_loss * mask).sum() / mask.sum()

    def compute_loss(self, batch, mode: str):
        model_output = self(**batch["net_input"])
        valid_mask = batch["valid_mask"]
        ss8_loss = nn.CrossEntropyLoss(ignore_index=-1)(
            model_output["Q8"].view(-1, 8), batch["Q8"]
        )
        ss3_loss = nn.CrossEntropyLoss(ignore_index=-1)(
            model_output["Q3"].view(-1, 3), batch["Q3"]
        )

        disorder_loss = self._masked_loss(
            nn.BCEWithLogitsLoss()(model_output["Disorder"], batch["Disorder"]),
            valid_mask,
        )

        phi_loss = self._masked_loss(
            nn.MSELoss(reduction="none")(model_output["Phi"], batch["Phi"]),
            batch["Disorder"],
            valid_mask,
        )
        psi_loss = self._masked_loss(
            nn.MSELoss(reduction="none")(model_output["Psi"], batch["Phi"]),
            batch["Disorder"],
            valid_mask,
        )
        rsa_loss = self._masked_loss(
            nn.MSELoss()(model_output["RSA"], batch["RSA"]),
            valid_mask,
        )

        interface_loss = self._masked_loss(
            nn.BCEWithLogitsLoss()(model_output["Interface"], batch["Interface"]),
            valid_mask,
        )

        total_loss = (
            ss8_loss
            + ss3_loss
            + disorder_loss
            + phi_loss
            + psi_loss
            + rsa_loss
            + interface_loss
        )

        ss8_acc = pl.metrics.classification.Accuracy()(
            model_output["Q8"][valid_mask].view(-1, 8), batch["Q8"][valid_mask]
        )
        ss3_acc = pl.metrics.classification.Accuracy()(
            model_output["Q3"][valid_mask].view(-1, 3), batch["Q3"][valid_mask]
        )

        self.log(f"ss8/{mode}", ss8_acc)
        self.log(f"ss3/{mode}", ss3_acc)
        return total_loss, model_output

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "valid")[0]

    def test_step(self, batch, batch_idx):
        return self.compute_loss(batch, "test")[0]

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
