import typing
import os
import logging
from pathlib import Path
import json
import itertools
from tqdm import tqdm
import argparse
import warnings
import pickle as pkl

import torch
import torch.nn as nn
from pytorch_transformers import WarmupLinearSchedule

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

from tape_pytorch.registry import registry
import tape_pytorch.training as training
import tape_pytorch.utils as utils

CallbackList = typing.Sequence[typing.Callable]
OutputDict = typing.Dict[str, typing.List[typing.Any]]


logger = logging.getLogger(__name__)
warnings.filterwarnings(  # Ignore pytorch warning about loss gathering
    'ignore', message='Was asked to gather along dimension 0', module='torch.nn.parallel')


def create_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Parent parser for tape functions',
                                     add_help=False)
    parser.add_argument('model_type', choices=list(registry.model_name_mapping.keys()),
                        help='Base model class to run')
    parser.add_argument('--model-config-file', default=None, type=utils.check_is_file,
                        help='Config file for model')
    parser.add_argument('--data-dir', default='./data', type=utils.check_is_dir,
                        help='Directory from which to load task data')
    parser.add_argument('--vocab-file', default='data/pfam.model', type=utils.check_is_file,
                        help='Pretrained tokenizer vocab file')
    parser.add_argument('--output-dir', default='./results', type=str)
    parser.add_argument('--no-cuda', action='store_true', help='CPU-only flag')
    parser.add_argument('--seed', default=42, type=int, help='Random seed to use')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank of process in distributed training. '
                             'Set by launch script.')
    parser.add_argument('--tokenizer', choices=['bpe', 'dummy'], default='bpe',
                        help='Tokenizes to use on the amino acid sequences')
    parser.add_argument('--num-workers', default=16, type=int,
                        help='Number of workers to use for multi-threaded data loading')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    return parser


def create_train_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Training on the TAPE datasets',
                                     parents=[base_parser])
    parser.add_argument('task', choices=list(registry.dataset_name_mapping.keys()),
                        help='TAPE Task to train/eval on')
    parser.add_argument('--learning-rate', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--batch-size', default=1024, type=int,
                        help='Batch size')
    parser.add_argument('--num-train-epochs', default=10, type=int,
                        help='Number of training epochs')
    parser.add_argument('--num-log-iter', default=20, type=int,
                        help='Number of training steps per log iteration')
    parser.add_argument('--fp16', action='store_true', help='Whether to use fp16 weights')
    parser.add_argument('--warmup-steps', default=10000, type=int,
                        help='Number of learning rate warmup steps')
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int,
                        help='Number of forward passes to make for each backwards pass')
    parser.add_argument('--loss-scale', default=0, type=int,
                        help='Loss scaling. Only used during fp16 training.')
    parser.add_argument('--max-grad-norm', default=1.0, type=float,
                        help='Maximum gradient norm')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Name to give to this experiment')
    parser.add_argument('--from-pretrained', default=None, type=utils.check_is_dir,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--log-dir', default='./logs', type=str)
    parser.add_argument('--no-eval', action='store_true',
                        help='Flag to not run eval pass. Useful for gridsearching.')
    return parser


def create_eval_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Eval on the TAPE Datasets',
                                     parents=[base_parser])
    parser.add_argument('task', choices=list(registry.dataset_name_mapping.keys()),
                        help='TAPE Task to train/eval on')
    parser.add_argument('from_pretrained', type=utils.check_is_dir,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--batch-size', default=1024, type=int,
                        help='Batch size')
    parser.add_argument('--save-callback', default=['save_predictions'],
                        help=f'Callbacks to use when saving. '
                             f'Choices: {list(registry.callback_name_mapping.keys())}',
                        nargs='*')
    parser.add_argument('--metrics', default=[],
                        help=f'Metrics to run on the result. '
                             f'Choices: {list(registry.metric_name_mapping.keys())}',
                        nargs='*')
    parser.add_argument('--split', default='test', type=str,
                        help='Which split to run on')
    return parser


def create_embed_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Embed a set of proteins wiht a pretrained model',
        parents=[base_parser])
    parser.add_argument('datafile', type=str,
                        help='File containing set of proteins to embed')
    parser.add_argument('outfile', type=str,
                        help='Name of output file')
    parser.add_argument('from_pretrained', type=utils.check_is_dir,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--batch-size', default=1024, type=int,
                        help='Batch size')
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

    device, n_gpu, is_master = utils.setup_distributed(args.local_rank, args.no_cuda)
    args.device = device
    args.n_gpu = n_gpu
    args.is_master = is_master

    save_path, exp_name = utils.get_savepath_and_expname(
        args.output_dir, args.exp_name, args.is_master)

    if args.is_master:
        # save all the hidden parameters.
        with (save_path / 'config.json').open('w') as f:
            json.dump({name: str(val) for name, val in vars(args).items()}, f)

    utils.setup_logging(args.local_rank, save_path)
    utils.set_random_seeds(args.seed, args.n_gpu)

    logger.info(
        f"device: {args.device} "
        f"n_gpu: {args.n_gpu}, "
        f"distributed_training: {args.local_rank != -1}, "
        f"16-bits training: {args.fp16}")

    model = utils.setup_model(
        args.task, args.from_pretrained, args.model_config_file, args.model_type)
    optimizer = utils.setup_optimizer(model, args.learning_rate)
    viz = utils.TBLogger(args.log_dir, exp_name, args.local_rank)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    if args.local_rank != -1:
        model = DDP(model)
    elif args.n_gpu > 1:
        model = nn.DataParallel(model)  # type: ignore

    train_dataset = utils.setup_dataset(args.task, args.data_dir, 'train', args.tokenizer)
    valid_dataset = utils.setup_dataset(args.task, args.data_dir, 'valid', args.tokenizer)
    train_loader = utils.setup_loader(
        args.task, train_dataset, args.batch_size, args.local_rank, args.n_gpu,
        args.gradient_accumulation_steps, args.num_workers)
    valid_loader = utils.setup_loader(
        args.task, valid_dataset, args.batch_size, args.local_rank, args.n_gpu,
        args.gradient_accumulation_steps, args.num_workers)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, args.batch_size, args.num_train_epochs)

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    trainer = training.BackwardRunner(
        model, optimizer, scheduler, args.device, args.n_gpu, args.fp16, args.max_grad_norm)

    training.run_train(
        model, trainer, train_dataset, train_loader, valid_dataset, valid_loader,
        viz, save_path, args.batch_size, args.num_train_epochs, args.local_rank, n_gpu,
        args.gradient_accumulation_steps, args.num_log_iter, args.no_eval, args.save_freq)


def run_eval(args: typing.Optional[argparse.Namespace] = None) -> typing.Dict[str, float]:
    if args is None:
        base_parser = create_base_parser()
        parser = create_eval_parser(base_parser)
        args = parser.parse_args()

    if args.from_pretrained is None:
        raise ValueError("Must specify pretrained model")
    if args.local_rank != -1:
        raise ValueError("TAPE does not support distributed validation pass")

    device, n_gpu, is_master = utils.setup_distributed(args.local_rank, args.no_cuda)
    args.device = device
    args.n_gpu = n_gpu
    args.is_master = is_master

    utils.setup_logging(args.local_rank, save_path=None)
    utils.set_random_seeds(args.seed, args.n_gpu)

    pretrained_dir = Path(args.from_pretrained)

    logger.info(
        f"device: {args.device} "
        f"n_gpu: {args.n_gpu}")

    model = utils.setup_model(
        args.task, args.from_pretrained, args.model_config_file, args.model_type)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)  # type: ignore

    runner = training.ForwardRunner(model, args.device, args.n_gpu)
    valid_dataset = utils.setup_dataset(args.task, args.data_dir, 'valid', args.tokenizer)
    valid_loader = utils.setup_loader(
        args.task, valid_dataset, args.batch_size, args.local_rank, args.n_gpu,
        1, args.num_workers)

    save_callbacks = [registry.get_callback(name) for name in args.save_callback]

    if len(args.metrics) > 0 and 'save_predictions' not in args.save_callback:
        save_callbacks.append(registry.get_callback('save_predictions'))
    metric_functions = [registry.get_metric(name) for name in args.metrics]

    save_outputs = training.run_eval_epoch(valid_loader, runner, args.is_master, save_callbacks)

    target_key = getattr(model, 'module', model).TARGET_KEY
    prediction_key = getattr(model, 'module', model).PREDICTION_KEY
    metrics = {name: metric(save_outputs[target_key], save_outputs[prediction_key])
               for name, metric in zip(args.metrics, metric_functions)}
    save_outputs.update(metrics)
    logger.info(f'Evaluation Metrics: {metrics}')

    with (pretrained_dir / 'results.pkl').open('wb') as f:
        pkl.dump(save_outputs, f)

    return metrics


def run_embed(args: typing.Optional[argparse.Namespace] = None) -> None:
    if args is None:
        base_parser = create_base_parser()
        parser = create_embed_parser(base_parser)
        args = parser.parse_args()

    if args.from_pretrained is None:
        raise ValueError("Must specify pretrained model")
    if args.local_rank != -1:
        raise ValueError("TAPE does not support distributed embed pass")

    device, n_gpu, is_master = utils.setup_distributed(args.local_rank, args.no_cuda)
    args.device = device
    args.n_gpu = n_gpu
    args.is_master = is_master

    utils.setup_logging(args.local_rank, save_path=None)
    utils.set_random_seeds(args.seed, args.n_gpu)

    logger.info(
        f"device: {args.device} "
        f"n_gpu: {args.n_gpu}")

    model = utils.setup_model(
        args.model_type, args.task, args.from_pretrained, args.model_config_file)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)  # type: ignore

    dataset = utils.setup_dataset(args.task, args.data_dir, args.datafile, args.tokenizer)
    loader = utils.setup_loader(
        args.task, dataset, args.batch_size, args.local_rank, args.n_gpu, 1, args.num_workers)

    torch.set_grad_enabled(False)
    model.eval()

    save_outputs = []
    save_callback = registry.get_callback('save_embedding')

    for batch in tqdm(loader, desc='Embedding sequences', total=len(loader),
                      disable=not args.is_master):
        cuda_batch = {name: tensor.cuda(device=args.device, non_blocking=True)
                      for name, tensor in batch.items()}
        outputs = model(**cuda_batch)

        to_save = save_callback(model, batch, outputs)
        save_outputs.append(to_save)

    keys = save_outputs[0].keys()
    output_dict = {
        key: list(itertools.chain.from_iterable(output[key] for output in save_outputs))
        for key in keys}

    with (Path(args.outfile).with_suffix('.pkl')).open('wb') as f:
        pkl.dump(output_dict, f)


def run_train_distributed(args: typing.Optional[argparse.Namespace] = None) -> None:
    """Runs distributed training via multiprocessing. Mostly ripped from
    pytorch's torch.distributed.launch, modified to be easy to use for
    tape.
    """
    from multiprocessing import Process
    import time

    if args is None:
        base_parser = create_base_parser()
        distributed_parser = create_distributed_parser(base_parser)
        distributed_train_parser = create_train_parser(distributed_parser)
        args = distributed_train_parser.parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************".format(
                  current_env["OMP_NUM_THREADS"]))

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        args.local_rank = local_rank
        process = Process(target=run_train, kwargs={'args': args, 'env': current_env})
        process.start()
        processes.append(process)

    while all(process.is_alive() for process in processes):
        time.sleep(1)

    process_failed = any(process.exitcode not in (None, 0) for process in processes)

    if process_failed:
        for process in processes:
            if process.is_alive():
                process.terminate()

    # for process in processes:
        # process.join()
        # if process.exitcode != 0:
            # raise ProcessError(f"Process failed with exitcode {process.exitcode}")


def run_gridsearch(args: typing.Optional[argparse.Namespace] = None, env=None) -> None:
    import random
    from itertools import product
    from copy import copy
    import shutil

    if env is not None:
        os.environ = env

    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_file', type=argparse.FileType('r'))
        gridsearch_args = parser.parse_args()
        config = json.load(gridsearch_args.config_file)
        gridsearch_args.config_file.close()
        # print(gridsearch_args.config_file)
        # with gridsearch_args.config_file.open() as f:
            # config = json.load(f)

        fixed_values = {}
        grid_values = {}

        for key, value in config.items():
            if isinstance(value, list) and key != 'metrics':
                grid_values[key] = value
            else:
                fixed_values[key] = value

        args = argparse.Namespace(**fixed_values)

    args.no_eval = True
    args.exp_name = 'gridsearch' + "_{:0>6d}".format(random.randint(0, int(1e6)))
    args.save_callback = []

    def unroll(key, values):
        return ((key, value) for value in values)

    results = []
    for grid_args in product(*(unroll(key, values) for key, values in grid_values.items())):
        run_args = copy(args)
        for key, arg in grid_args:
            setattr(run_args, key, arg)
        run_train(copy(run_args))
        run_args.from_pretrained = os.path.join(
            run_args.output_dir, run_args.exp_name,
            f'pytorch_model_{run_args.num_train_epochs - 1}')
        metrics = run_eval(copy(run_args))
        results.append((grid_args, metrics))
        shutil.rmtree(run_args.from_pretrained)

    for grid_args, metrics in results:
        print(grid_args)
        print(metrics)
        print()

    with open(os.path.join(
            args.output_dir, args.exp_name, 'gridsearch_results.pkl'), 'wb') as f:
        pkl.dump(results, f)


if __name__ == '__main__':
    # run_train()
    run_embed()
