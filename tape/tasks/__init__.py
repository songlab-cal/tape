from typing import Tuple
from argparse import Namespace
import torch.nn as nn
from .tape_task import TAPEDataModule, TAPEPredictorBase


def get(
    args: Namespace, base_model: nn.Module, extract_features, embedding_dim: int
) -> Tuple[TAPEDataModule, TAPEPredictorBase]:
    if args.task == "secondary_structure":
        from .secondary_structure_task import (
            SecondaryStructureDatamodule,
            SecondaryStructurePredictor,
        )

        task_data = SecondaryStructureDatamodule(
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
    elif args.task == "fluorescence":
        from .fluorescence_task import (
            FluorescenceDataModule,
            FluorescencePredictor,
        )

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
