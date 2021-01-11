from typing import Union, Dict, List, Callable, Optional
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import pytorch_lightning as pl

from tape.tokenizers import TAPETokenizer
from ..utils import pad_sequences, PathLike, TensorDict
from .tape_task import (
    TAPEDataset,
    TAPEDataModule,
    TAPEPredictorBase,
    TAPETask,
)


class SecondaryStructureDataset(TAPEDataset):
    def __init__(
        self,
        data_path: PathLike,
        split: str,
        tokenizer: Union[str, TAPETokenizer] = "iupac",
        use_msa: bool = False,
        max_tokens_per_msa: int = 2 ** 14,
    ):
        super().__init__(
            data_path=data_path,
            split=split,
            tokenizer=tokenizer,
            use_msa=use_msa,
            max_tokens_per_msa=max_tokens_per_msa,
        )

    @property
    def task_name(self) -> str:
        return "secondary_structure"

    @property
    def splits(self) -> List[str]:
        return ["train", "valid", "cb513", "ts115", "casp12"]

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

    def resort_result(
        self, result: Dict[str, Union[torch.Tensor, TensorDict]], indices: torch.Tensor
    ) -> Dict[str, Union[torch.Tensor, TensorDict]]:
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                result[key] = value[indices]
            else:
                result[key] = self.resort_result(value, indices)  # type: ignore
        return result

    def collate_fn(
        self, batch: List[TensorDict]
    ) -> Dict[str, Union[torch.Tensor, TensorDict]]:
        batch_size = len(batch)
        src_tokens = pad_sequences([el["src_tokens"] for el in batch])
        src_lengths = torch.tensor(
            [len(el["src_tokens"]) for el in batch], dtype=torch.long
        )

        result = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
            "Q8": pad_sequences([el["Q8"] for el in batch], constant_value=-1),
            "Q3": pad_sequences([el["Q3"] for el in batch], constant_value=-1),
            "RSA": pad_sequences([el["RSA"] for el in batch]).view(batch_size, -1, 1),
            "Phi": pad_sequences([el["Phi"] for el in batch]),
            "Psi": pad_sequences([el["Psi"] for el in batch]),
            "Disorder": pad_sequences([el["Disorder"] for el in batch]).view(
                batch_size, -1, 1
            ),
            "Interface": pad_sequences([el["Interface"] for el in batch]).view(
                batch_size, -1, 1
            ),
            "valid_mask": pad_sequences([el["valid_mask"] for el in batch]),
        }
        result = self.resort_result(result, src_lengths.argsort(descending=True))
        return result


class SecondaryStructureDataModule(TAPEDataModule):
    @property
    def data_url(self) -> str:
        return "http://s3.amazonaws.com/proteindata/data_pytorch/secondary_structure.tar.gz"  # noqa: E501

    @property
    def task_name(self) -> str:
        return "secondary_structure"

    def train_dataloader(self):
        dataset = SecondaryStructureDataset(self.data_dir, "train", self.tokenizer)
        return self.make_dataloader(dataset, shuffle=True)

    def val_dataloader(self):
        dataset = SecondaryStructureDataset(self.data_dir, "valid", self.tokenizer)
        return self.make_dataloader(dataset, shuffle=False)

    def test_dataloader(self):
        return [
            self.make_dataloader(
                SecondaryStructureDataset(self.data_dir, test_set, self.tokenizer),
                shuffle=False,
            )
            for test_set in ["cb513", "ts115", "casp12"]
        ]


class SecondaryStructurePredictor(TAPEPredictorBase):
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

        self.ss3_train_accuracy = pl.metrics.classification.Accuracy()
        self.ss8_train_accuracy = pl.metrics.classification.Accuracy()
        self.ss3_valid_accuracy = pl.metrics.classification.Accuracy()
        self.ss8_valid_accuracy = pl.metrics.classification.Accuracy()
        self.ss3_test_accuracy = pl.metrics.classification.Accuracy()
        self.ss8_test_accuracy = pl.metrics.classification.Accuracy()

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

    @classmethod
    def from_argparse_args(
        cls,
        args: Namespace,
        base_model: nn.Module,
        extract_features: Callable[
            [nn.Module, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
        ],
        embedding_dim: int,
    ):
        return cls(
            base_model=base_model,
            extract_features=extract_features,
            embedding_dim=embedding_dim,
            freeze_base=args.freeze_base,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler=args.lr_scheduler,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            conv_dropout=args.conv_dropout,
            lstm_dropout=args.lstm_dropout,
        )

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

    def _masked_loss(self, unreduced_loss, *masks):
        dtype = unreduced_loss.dtype
        mask = masks[0].to(dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(2)
        for addmask in masks[1:]:
            if addmask.dim() == 2:
                addmask = addmask.unsqueeze(2)
            mask = mask * addmask.to(dtype)

        return (unreduced_loss * mask).sum() / mask.sum()

    def compute_loss(self, batch, mode: str):
        model_output = self(**batch["net_input"])
        valid_mask = batch["valid_mask"]
        if model_output["Q8"].size()[:2] != batch["Q8"].size():
            raise RuntimeError(
                "Model output does not match batch size. Make sure "
                "extract_features(...) removes extra tokesns (e.g. start / end tokens)."
                f"\n\tModel output: {model_output['Q8'].size()[:2]}, "
                f"Expected {batch['Q8'].size()}."
            )
        ss8_loss = nn.CrossEntropyLoss(ignore_index=-1)(
            model_output["Q8"].view(-1, 8),
            batch["Q8"].view(-1),
        )
        ss3_loss = nn.CrossEntropyLoss(ignore_index=-1)(
            model_output["Q3"].view(-1, 3),
            batch["Q3"].view(-1),
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

        self.log(f"loss/{mode}", total_loss)
        model_output["loss"] = total_loss
        model_output["target"] = batch
        return model_output

    def compute_and_log_accuracy(self, outputs, mode: str):
        valid_mask = outputs["target"]["valid_mask"]
        ss3_accuracy = getattr(self, f"ss3_{mode}_accuracy")
        ss8_accuracy = getattr(self, f"ss8_{mode}_accuracy")
        ss3_accuracy(
            outputs["Q3"][valid_mask].view(-1, 3), outputs["target"]["Q3"][valid_mask]
        )
        ss8_accuracy(
            outputs["Q8"][valid_mask].view(-1, 8), outputs["target"]["Q8"][valid_mask]
        )
        self.log(f"ss3_acc/{mode}", ss3_accuracy)
        self.log(f"ss8_acc/{mode}", ss8_accuracy)

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, "train")

    def training_step_end(self, outputs):
        self.compute_and_log_accuracy(outputs, "train")
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "valid")

    def validation_step_end(self, outputs):
        self.compute_and_log_accuracy(outputs, "valid")
        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        return self.compute_loss(batch, "test")

    def test_step_end(self, outputs):
        self.compute_and_log_accuracy(outputs, "test")
        return outputs["loss"]


SecondaryStructureTask = TAPETask(
    "secondary_structure",
    SecondaryStructureDataModule,  # type: ignore
    SecondaryStructurePredictor,
)
