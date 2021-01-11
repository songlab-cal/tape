from argparse import ArgumentParser, Namespace
import torch
import sys
import logging
from tape import utils
from tape import tasks

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)


def extract_features(model, src_tokens, src_lengths):
    return model(src_tokens, utils.seqlen_mask(src_tokens, src_lengths))[0][:, 1:-1]


def get_num_gpus(args: Namespace) -> int:
    try:
        return int(args.gpus)
    except ValueError:
        return len(args.gpus.split(","))


def maybe_unset_distributed(args: Namespace) -> None:
    if get_num_gpus(args) <= 1:
        args.distributed_backend = None


def train():
    import pytorch_lightning as pl
    from tape.models.modeling_bert import ProteinBertModel
    from tape.tasks import SecondaryStructureDatamodule, SecondaryStructurePrediction
    # Initialize parser
    parser = ArgumentParser()
    parser.add_argument(
        "task",
        choices=["secondary_structure", "fluorescence"],
        help="Which downstream task to train."
    )
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
    maybe_unset_distributed(args)

    data = SecondaryStructureDatamodule(
        args.data_dir, args.batch_size, args.num_workers
    )
    base_model = ProteinBertModel.from_pretrained("bert-base")

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
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(task_model, data)
    trainer.teset(task_model, data)


if __name__ == "__main__":
    train()
