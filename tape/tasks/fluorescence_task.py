from typing import Union, Dict, List, Callable, Optional
from pathlib import Path
from argparse import ArgumentParser

import scipy.stats
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset
import pytorch_lightning as pl

from tape.datasets import LMDBDataset
from tape.tokenizers import TAPETokenizer
from ..utils import pad_sequences, PathLike, TensorDict
from .tape_task import TAPEDataModule, TAPEPredictorBase, DEFAULT_DATA_DIR


class FluorescenceDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str,
        tokenizer: Union[str, TAPETokenizer] = "iupac",
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        src_tokens = torch.from_numpy(self.tokenizer.encode(item["primary"])).long()
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
        return result


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
        conv_dropout: float = 0.1,
        lstm_dropout: float = 0.1,
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
            "conv_dropout",
            "lstm_dropout",
        )

        output_names = ["Q8", "Q3", "RSA", "Phi", "Psi", "Disorder", "Interface"]
        output_sizes = [8, 3, 1, 2, 2, 1, 1]

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

        self.mean_squared_error = pl.metrics.MeanSquaredError()
        self.mean_absolute_error = pl.metrics.MeanAbsoluteError()

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser = TAPEPredictorBase.add_args(parser)
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
        return parser

    def forward(self, src_tokens, src_lengths):
        # B x L x D
        features = self.extract_features(src_tokens, src_lengths)
        num_removed_tokens = src_tokens.size(1) - features.size(1)
        # B x L x D -> B x D x L
        features = features.transpose(1, 2)
        conv1_out = self.conv1(features)
        conv2_out = self.conv2(features)
        features = torch.cat([features, conv1_out, conv2_out], 1)
        # B x D x L -> L x B x D
        features = features.permute(2, 0, 1)
        lstm_input = rnn.pack_padded_sequence(
            features, src_lengths - num_removed_tokens
        )
        lstm_output, _ = self.lstm(lstm_input)
        # L x B x D -> B x L x D
        lstm_output, _ = rnn.pad_packed_sequence(lstm_output)
        lstm_output = lstm_output.transpose(0, 1)

        output = self.outproj(lstm_output)
        output = output.split(self.output_sizes, dim=-1)
        return dict(zip(self.output_names, output))

    def compute_loss(self, batch, mode: str):
        log_fluorescence = self(**batch["net_input"])
        loss = (nn.MSELoss()(log_fluorescence, batch["log_fluorescence"]),)
        self.log(f"loss/{mode}", loss)

        return {"loss": loss, "log_fluorescence": log_fluorescence, "target": batch}

    def compute_and_log_accuracy(self, outputs, mode: str):
        self.mean_squared_error(
            outputs["log_fluorescence"], outputs["target"]["log_fluorescence"]
        )
        self.mean_absolute_error(
            outputs["log_fluorescence"], outputs["target"]["log_fluorescence"]
        )

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
        predictions = torch.stack([step["log_fluorescence"] for step in outputs], 0)
        targets = torch.stack(
            [step["target"]["log_fluorescence"] for step in outputs], 0
        )
        corr, _ = scipy.stats.spearmanr(predictions, targets)
        self.log("spearmanr/valid", corr)

    def test_step(self, batch, batch_idx):
        return self.compute_loss(batch, "test")

    def test_step_end(self, outputs):
        self.compute_and_log_accuracy(outputs, "test")
        return outputs

    def test_epoch_end(self, outputs):
        predictions = torch.stack([step["log_fluorescence"] for step in outputs], 0)
        targets = torch.stack(
            [step["target"]["log_fluorescence"] for step in outputs], 0
        )
        corr, _ = scipy.stats.spearmanr(predictions, targets)
        self.log("spearmanr/test", corr)
