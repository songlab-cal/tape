from argparse import ArgumentParser
import sys
import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)


def train():
    import pytorch_lightning as pl
    from tape.models.modeling_bert import ProteinBertModel
    from tape.tasks import SecondaryStructureDatamodule, SecondaryStructurePrediction
    # Initialize parser
    parser = ArgumentParser()
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Optional wandb project to log to.",
    )
    parser = SecondaryStructureDatamodule.add_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SecondaryStructurePrediction.add_args(parser)
    parser.set_defaults(
        gpus=1,
        min_steps=50,
        max_steps=1000,
    )
    args = parser.parse_args()

    data = SecondaryStructureDatamodule(
        args.data_dir, args.batch_size, args.num_workers
    )
    base_model = ProteinBertModel.from_pretrained("bert-base")
    task_model = SecondaryStructurePrediction(
        base_model,
        768,
        args.conv_dropout,
        args.lstm_dropout,
        args.freeze_base,
        args.optimizer,
        args.learning_rate,
        args.weight_decay,
        args.lr_scheduler,
        args.warmup_steps,
        args.max_steps,
    )

    kwargs = {}
    if args.wandb_project:
        try:
            # Requires wandb to be installed
            logger = pl.loggers.WandbLogger(project=args.wandb_project)
            logger.log_hyperparams(args)
            kwargs["logger"] = logger
        except ImportError:
            raise ImportError(
                "Cannot use W&B logger w/o W&b install. Run `pip install wandb` first."
            )

    # Initialize Trainer
    trainer = pl.Trainer.from_argparse_args(args, **kwargs)
    trainer.fit(task_model, data)
    trainer.teset(task_model, data)


if __name__ == "__main__":
    train()
