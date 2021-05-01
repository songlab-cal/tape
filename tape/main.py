import typing
import os
import logging
import argparse
import warnings
import inspect


try:
    import apex  # noqa: F401
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

from .registry import registry
from . import training
from . import utils

CallbackList = typing.Sequence[typing.Callable]
OutputDict = typing.Dict[str, typing.List[typing.Any]]


logger = logging.getLogger(__name__)
warnings.filterwarnings(  # Ignore pytorch warning about loss gathering
    'ignore', message='Was asked to gather along dimension 0', module='torch.nn.parallel')


def create_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Parent parser for tape functions',
                                     add_help=False)
    parser.add_argument('model_type', help='Base model class to run')
    parser.add_argument('--model_config_file', default=None, type=utils.check_is_file,
                        help='Config file for model')
    parser.add_argument('--vocab_file', default=None,
                        help='Pretrained tokenizer vocab file')
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='CPU-only flag')
    parser.add_argument('--seed', default=42, type=int, help='Random seed to use')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank of process in distributed training. '
                             'Set by launch script.')
    parser.add_argument('--tokenizer', choices=['iupac', 'unirep'],
                        default='iupac', help='Tokenizes to use on the amino acid sequences')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers to use for multi-threaded data loading')
    parser.add_argument('--log_level', default=logging.INFO,
                        choices=['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR',
                                 logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
                        help="log level for the experiment")
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    return parser


def create_train_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Training on the TAPE datasets',
                                     parents=[base_parser])
    parser.add_argument('task', choices=list(registry.task_name_mapping.keys()),
                        help='TAPE Task to train/eval on')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size')
    parser.add_argument('--data_dir', default='./data', type=utils.check_is_dir,
                        help='Directory from which to load task data')
    parser.add_argument('--num_train_epochs', default=10, type=int,
                        help='Number of training epochs')
    parser.add_argument('--num_log_iter', default=20, type=int,
                        help='Number of training steps per log iteration')
    parser.add_argument('--fp16', action='store_true', help='Whether to use fp16 weights')
    parser.add_argument('--warmup_steps', default=10000, type=int,
                        help='Number of learning rate warmup steps')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of forward passes to make for each backwards pass')
    parser.add_argument('--loss_scale', default=0, type=int,
                        help='Loss scaling. Only used during fp16 training.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Maximum gradient norm')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Name to give to this experiment')
    parser.add_argument('--from_pretrained', default=None, type=str,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--log_dir', default='./logs', type=str)
    parser.add_argument('--eval_freq', type=int, default=1,
                        help="Frequency of eval pass. A value <= 0 means the eval pass is "
                             "not run")
    parser.add_argument('--save_freq', default=1, type=utils.int_or_str,
                        help="How often to save the model during training. Either an integer "
                             "frequency or the string 'improvement'")
    parser.add_argument('--patience', default=-1, type=int,
                        help="How many epochs without improvement to wait before ending "
                             "training")
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                        help="whether to resume training from the checkpoint")
    return parser


def create_eval_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Eval on the TAPE Datasets',
                                     parents=[base_parser])
    parser.add_argument('task', choices=list(registry.task_name_mapping.keys()),
                        help='TAPE Task to train/eval on')
    parser.add_argument('from_pretrained', type=str,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size')
    parser.add_argument('--data_dir', default='./data', type=utils.check_is_dir,
                        help='Directory from which to load task data')
    parser.add_argument('--metrics', default=[],
                        help=f'Metrics to run on the result. '
                             f'Choices: {list(registry.metric_name_mapping.keys())}',
                        nargs='*')
    parser.add_argument('--split', default='test', type=str,
                        help='Which split to run on')
    return parser


def create_embed_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Embed a set of proteins with a pretrained model',
        parents=[base_parser])
    parser.add_argument('data_file', type=str,
                        help='File containing set of proteins to embed')
    parser.add_argument('out_file', type=str,
                        help='Name of output file')
    parser.add_argument('from_pretrained', type=str,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size')
    parser.add_argument('--full_sequence_embed', action='store_true',
                        help='If true, saves an embedding at every amino acid position '
                             'in the sequence. Note that this can take a large amount '
                             'of disk space.')
    parser.set_defaults(task='embed')
    return parser


def create_distributed_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, parents=[base_parser])
    # typing.Optional arguments for the launch helper
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")
    return parser


def run_train(args: typing.Optional[argparse.Namespace] = None, env=None) -> None:
    if env is not None:
        os.environ = env

    if args is None:
        base_parser = create_base_parser()
        train_parser = create_train_parser(base_parser)
        args = train_parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            f"Invalid gradient_accumulation_steps parameter: "
            f"{args.gradient_accumulation_steps}, should be >= 1")

    if (args.fp16 or args.local_rank != -1) and not APEX_FOUND:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex "
            "to use distributed and fp16 training.")

    arg_dict = vars(args)
    arg_names = inspect.getfullargspec(training.run_train).args

    missing = set(arg_names) - set(arg_dict.keys())
    if missing:
        raise RuntimeError(f"Missing arguments: {missing}")
    train_args = {name: arg_dict[name] for name in arg_names}

    training.run_train(**train_args)


def run_eval(args: typing.Optional[argparse.Namespace] = None) -> typing.Dict[str, float]:
    if args is None:
        base_parser = create_base_parser()
        parser = create_eval_parser(base_parser)
        args = parser.parse_args()

    if args.from_pretrained is None:
        raise ValueError("Must specify pretrained model")
    if args.local_rank != -1:
        raise ValueError("TAPE does not support distributed validation pass")

    arg_dict = vars(args)
    arg_names = inspect.getfullargspec(training.run_eval).args

    missing = set(arg_names) - set(arg_dict.keys())
    if missing:
        raise RuntimeError(f"Missing arguments: {missing}")
    eval_args = {name: arg_dict[name] for name in arg_names}

    return training.run_eval(**eval_args)


def run_embed(args: typing.Optional[argparse.Namespace] = None) -> None:
    if args is None:
        base_parser = create_base_parser()
        parser = create_embed_parser(base_parser)
        args = parser.parse_args()
    if args.from_pretrained is None:
        raise ValueError("Must specify pretrained model")
    if args.local_rank != -1:
        raise ValueError("TAPE does not support distributed validation pass")

    arg_dict = vars(args)
    arg_names = inspect.getfullargspec(training.run_embed).args

    missing = set(arg_names) - set(arg_dict.keys())
    if missing:
        raise RuntimeError(f"Missing arguments: {missing}")
    embed_args = {name: arg_dict[name] for name in arg_names}

    training.run_embed(**embed_args)


def run_train_distributed(args: typing.Optional[argparse.Namespace] = None) -> None:
    """Runs distributed training via multiprocessing.
    """
    if args is None:
        base_parser = create_base_parser()
        distributed_parser = create_distributed_parser(base_parser)
        distributed_train_parser = create_train_parser(distributed_parser)
        args = distributed_train_parser.parse_args()

    # Define the experiment name here, instead of dealing with barriers and communication
    # when getting the experiment name
    exp_name = utils.get_expname(args.exp_name, args.task, args.model_type)
    args.exp_name = exp_name
    utils.launch_process_group(
        run_train, args, args.nproc_per_node, args.nnodes,
        args.node_rank, args.master_addr, args.master_port)


if __name__ == '__main__':
    run_train_distributed()
