from typing import Tuple
from argparse import ArgumentParser
import torch.nn as nn
from .tape_task import TAPEDataModule, TAPEPredictorBase


def get(
    parser: ArgumentParser, base_model: nn.Module, extract_features, embedding_dim: int
) -> Tuple[TAPEDataModule, TAPEPredictorBase]:
    known_args, _ = parser.parse_known_args()
    task_name = parser.parse_known_args()[0].task
    if task_name == "secondary_structure":
        from .secondary_structure_task import (
            SecondaryStructureDataModule,
            SecondaryStructurePredictor,
        )
        SecondaryStructureDataModule.add_args(parser)
        SecondaryStructurePredictor.add_args(parser)
        args = parser.parse_args()

        task_data = SecondaryStructureDataModule(
            args.data_dir, args.batch_size, args.num_workers
        )
        task_model = SecondaryStructurePredictor(
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
        return task_data, task_model
    elif task_name == "fluorescence":
        from .fluorescence_task import (
            FluorescenceDataModule,
            FluorescencePredictor,
        )

        FluorescenceDataModule.add_args(parser)
        FluorescencePredictor.add_args(parser)
        args = parser.parse_args()
        task_data = FluorescenceDataModule(
            args.data_dir, args.batch_size, args.num_workers
        )
        task_model = FluorescencePredictor(
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
            dropout=args.dropout,
            hidden_size=args.hidden_size,
        )
        return task_data, task_model
    else:
        raise ValueError(f"Unrecognized task {args.task}")
