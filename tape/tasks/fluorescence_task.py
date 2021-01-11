from typing import Union, Dict, List, Callable, Optional
import math
from pathlib import Path
from argparse import ArgumentParser

import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl

from tape.datasets import LMDBDataset
from tape.tokenizers import TAPETokenizer
from ..utils import (
    pad_sequences,
    PathLike,
    TensorDict,
    seqlen_mask,
    parse_fasta,
    hhfilter_sequences,
)
from .tape_task import TAPEDataModule, TAPEPredictorBase, DEFAULT_DATA_DIR


class FluorescenceDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str,
        tokenizer: Union[str, TAPETokenizer] = "iupac",
        use_msa: bool = False,
        max_tokens_per_msa: int = 2 ** 14,
    ):

        if split not in ("train", "valid", "test"):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'test']"
            )
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f"fluorescence/fluorescence_{split}.lmdb"
        self.data = LMDBDataset(data_path / data_file)
        self.use_msa = use_msa

        if use_msa:
            msa_file = "fluorescence/wtGFP.a3m"
            msa_path = data_path / msa_file
            _, msa = parse_fasta(msa_path, remove_insertions=True)

            seqlen = len(self.tokenizer.encode(self.data[0]["primary"]))
            max_num_sequences = max_tokens_per_msa // seqlen
            sequences_from_msa = max_num_sequences - 1

            sequences = hhfilter_sequences(msa, diff=sequences_from_msa)
            tokens = torch.stack(
                [
                    self.tokenizer.encode(seq)
                    for _, seq in sequences[:sequences_from_msa]
                ],
                0,
            ).long()
            self.msa = tokens

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        src_tokens = torch.from_numpy(self.tokenizer.encode(item["primary"])).long()
        if self.use_msa:
            src_tokens = src_tokens.unsqueeze(0)
            src_tokens = torch.cat([src_tokens, self.msa], 0)
        value = torch.tensor(item["log_fluorescence"][0], dtype=torch.float)
        return {"src_tokens": src_tokens, "log_fluorescence": value}

    def collate_fn(
        self, batch: List[TensorDict]
    ) -> Dict[str, Union[torch.Tensor, TensorDict]]:
        src_tokens = pad_sequences([el["src_tokens"] for el in batch])
        src_lengths = torch.tensor(
            [len(el["src_tokens"]) for el in batch], dtype=torch.long
        )

        result = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
            "log_fluorescence": torch.stack(
                [item["log_fluorescence"] for item in batch], 0
            ).unsqueeze(1),
        }
        return result  # type: ignore


class FluorescenceDataModule(TAPEDataModule):
    def __init__(
        self,
        data_dir: PathLike = DEFAULT_DATA_DIR,
        batch_size: int = 64,
        num_workers: int = 3,
        tokenizer: str = "iupac",
    ):
        super().__init__(
            data_url="http://s3.amazonaws.com/proteindata/data_pytorch/fluorescence.tar.gz",  # noqa: E501
            task_name="fluorescence",
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            tokenizer=tokenizer,
        )

    def train_dataloader(self):
        dataset = FluorescenceDataset(self.data_dir, "train", self.tokenizer)
        return self.make_dataloader(dataset, shuffle=True)

    def val_dataloader(self):
        dataset = FluorescenceDataset(self.data_dir, "valid", self.tokenizer)
        return self.make_dataloader(dataset, shuffle=False)

    def test_dataloader(self):
        dataset = FluorescenceDataset(self.data_dir, "test", self.tokenizer)
        return self.make_dataloader(dataset, shuffle=False)


class FluorescencePredictor(TAPEPredictorBase):
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
        dropout: float = 0.1,
        hidden_size: int = 512,
    ):
        super().__init__(
            base_model=base_model,
            extract_features=extract_features,
            embedding_dim=embedding_dim,
            freeze_base=freeze_base,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
        )
        self.save_hyperparameters(
            "dropout",
            "hidden_size",
        )

        self.compute_attention_weights = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.mean_train_squared_error = pl.metrics.MeanSquaredError()
        self.mean_train_absolute_error = pl.metrics.MeanAbsoluteError()
        self.mean_valid_squared_error = pl.metrics.MeanSquaredError()
        self.mean_valid_absolute_error = pl.metrics.MeanAbsoluteError()
        self.mean_test_squared_error = pl.metrics.MeanSquaredError()
        self.mean_test_absolute_error = pl.metrics.MeanAbsoluteError()

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser = TAPEPredictorBase.add_args(parser)
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout on attention block.",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=512,
            help="MLP hidden dimension.",
        )
        return parser

    def forward(self, src_tokens, src_lengths):
        # B x L x D
        features = self.extract_features(src_tokens, src_lengths)
        attention_weights = self.compute_attention_weights(features)
        attention_weights /= math.sqrt(features.size(2))
        mask = seqlen_mask(features, src_lengths - 2)
        attention_weights = attention_weights.masked_fill(~mask.unsqueeze(2), -10000)
        attention_weights = attention_weights.softmax(1)
        attention_weights = self.dropout(attention_weights)
        pooled_features = features.transpose(1, 2) @ attention_weights
        pooled_features = pooled_features.squeeze(2)
        return self.mlp(pooled_features)

    def compute_loss(self, batch, mode: str):
        log_fluorescence = self(**batch["net_input"])
        loss = nn.MSELoss()(log_fluorescence, batch["log_fluorescence"])
        self.log(f"loss/{mode}", loss)

        return {"loss": loss, "log_fluorescence": log_fluorescence, "target": batch}

    def compute_and_log_accuracy(self, outputs, mode: str):
        mse = getattr(self, f"mean_{mode}_squared_error")
        mae = getattr(self, f"mean_{mode}_absolute_error")
        mse(outputs["log_fluorescence"], outputs["target"]["log_fluorescence"])
        mae(outputs["log_fluorescence"], outputs["target"]["log_fluorescence"])
        self.log(f"mse/{mode}", mse)
        self.log(f"mae/{mode}", mae)

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, "train")

    def training_step_end(self, outputs):
        self.compute_and_log_accuracy(outputs, "train")
        return outputs

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "valid")

    def validation_step_end(self, outputs):
        self.compute_and_log_accuracy(outputs, "valid")
        return outputs

    def validation_epoch_end(self, outputs):
        predictions = torch.cat([step["log_fluorescence"] for step in outputs], 0)
        targets = torch.cat([step["target"]["log_fluorescence"] for step in outputs], 0)
        corr, _ = scipy.stats.spearmanr(predictions.cpu(), targets.cpu())
        self.log("spearmanr/valid", corr)

    def test_step(self, batch, batch_idx):
        return self.compute_loss(batch, "test")

    def test_step_end(self, outputs):
        self.compute_and_log_accuracy(outputs, "test")
        return outputs

    def test_epoch_end(self, outputs):
        predictions = torch.cat([step["log_fluorescence"] for step in outputs], 0)
        targets = torch.cat([step["target"]["log_fluorescence"] for step in outputs], 0)
        corr, _ = scipy.stats.spearmanr(predictions.cpu(), targets.cpu())
        self.log("spearmanr/test", corr)
