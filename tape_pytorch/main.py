from typing import Optional, Tuple, Type, Callable, Sequence, List, Dict, Any
import os
from time import strftime, gmtime
from timeit import default_timer as timer
import logging
from pathlib import Path
import json
import itertools
from tqdm import tqdm
import argparse
import warnings
from collections import ChainMap
import pickle as pkl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from pytorch_transformers.modeling_utils import PreTrainedModel
from pytorch_transformers import AdamW
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
from tape_pytorch.datasets import TAPEDataset


logger = logging.getLogger(__name__)
warnings.filterwarnings(  # Ignore pytorch warning about loss gathering
    'ignore', message='Was asked to gather along dimension 0', module='torch.nn.parallel')


def setup_optimizer(model: PreTrainedModel,
                    learning_rate: float):

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    return optimizer


def setup_distributed(args: argparse.Namespace) -> argparse.Namespace:
    if args.local_rank != -1 and not args.no_cuda:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        dist.init_process_group(backend="nccl")
    elif not torch.cuda.is_available() or args.no_cuda:
        args.device = torch.device("cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()

    args.is_master = args.local_rank in (-1, 0)

    return args


def setup_dataset_and_loader(args: argparse.Namespace,
                             dataset_type: str) -> Tuple[TAPEDataset, DataLoader]:

    dataset_class: Type[TAPEDataset] = registry.get_dataset_class(  # type: ignore
        args.task)
    collate_fn_cls = registry.get_collate_fn_class(args.task)
    sampler_type = (DistributedSampler if args.local_rank != -1 else RandomSampler)

    dataset = dataset_class(args.data_dir, dataset_type, args.tokenizer)

    loader = DataLoader(  # type: ignore
        dataset,
        batch_size=utils.get_effective_batch_size(
            args.batch_size, args.local_rank, args.n_gpu,
            getattr(args, 'gradient_accumulation_steps', 1)),
        num_workers=args.num_workers,
        collate_fn=collate_fn_cls(),
        sampler=sampler_type(dataset))

    return dataset, loader


def get_num_train_optimization_steps(train_dataset: TAPEDataset,
                                     args: argparse.Namespace) -> int:
    return int(len(train_dataset) / args.batch_size * args.num_train_epochs)


def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    trainer: training.Trainer,
                    viz: utils.TBLogger,
                    args: argparse.Namespace) -> None:
    metrics = utils.MetricsAccumulator(smoothing=1 - 1 / args.num_log_iter)

    torch.set_grad_enabled(True)
    trainer.model.train()

    start_t = timer()

    if args.debug:
        train_loader = itertools.islice(train_loader, 10)  # type: ignore

    loss_tmp = 0.
    for step, batch in enumerate(train_loader):
        loss = trainer.forward(batch)
        loss /= args.gradient_accumulation_steps
        loss_tmp += loss.item()
        trainer.backward(loss)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            trainer.step()
            metrics.update(loss_tmp)
            viz.line_plot(trainer.global_step, loss_tmp, "loss", "train")
            loss_tmp = 0.

            if trainer.global_step % args.num_log_iter == 0:
                end_t = timer()
                time_stamp = strftime("%y-%m-%d %X", gmtime())
                ep = epoch_id + step / float(len(train_loader))
                print_str = [
                    f"[{time_stamp}]",
                    f"[Ep: {ep:.2f}]",
                    f"[Iter: {trainer.global_step}]",
                    f"[Time: {end_t - start_t:5.2f}s]",
                    f"[Loss: {metrics.loss():.5g}]",
                    f"[LR: {trainer.scheduler.get_lr()[0]:.5g}]"]  # type: ignore
                start_t = end_t

                logger.info(''.join(print_str))

    logger.info(f"Train: [Loss: {metrics.final_loss():.5g}]")


def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    trainer: training.Trainer,
                    viz: utils.TBLogger,
                    args: argparse.Namespace) -> None:
    num_batches = len(valid_loader)
    eval_loss = 0.

    torch.set_grad_enabled(False)
    trainer.model.eval()

    if args.debug:
        valid_loader = itertools.islice(valid_loader, 10)  # type: ignore

    for batch in tqdm(valid_loader, desc='Evaluating split val', total=num_batches,
                      disable=not args.is_master):
        loss = trainer.forward(batch)
        eval_loss += loss.item()

    eval_loss /= num_batches

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"

    logger.info(print_str)
    viz.line_plot(epoch_id, eval_loss, "loss", "val")


def run_eval_epoch(eval_loader: DataLoader,
                   model: nn.Module,
                   args: argparse.Namespace,
                   save_callback: Optional[Sequence[Callable]] = None) \
        -> Dict[str, List[Any]]:
    num_batches = len(eval_loader)
    eval_loss = 0.
    loss_key = getattr(model, 'module', model).LOSS_KEY

    torch.set_grad_enabled(False)
    model.eval()

    save_outputs = []

    for batch in tqdm(eval_loader, desc='Evaluating split val', total=num_batches,
                      disable=not args.is_master):
        cuda_batch = {name: tensor.cuda(device=args.device, non_blocking=True)
                      for name, tensor in batch.items()}
        outputs = model(**cuda_batch)
        loss = outputs[loss_key]

        if args.n_gpu > 1:
            loss = loss.mean()

        if save_callback is not None:
            to_save = dict(ChainMap(
                *(callback(model, batch, outputs) for callback in save_callback)))
            save_outputs.append(to_save)

        eval_loss += loss.item()

    eval_loss /= num_batches

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"

    logger.info(print_str)

    if len(save_outputs) > 0:
        keys = save_outputs[0].keys()
        output_dict = {
            key: list(itertools.chain.from_iterable(output[key] for output in save_outputs))
            for key in keys}
    else:
        output_dict = {}

    return output_dict


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
    # Optional arguments for the launch helper
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


def run_train(args: Optional[argparse.Namespace] = None, env=None) -> None:
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

    args = setup_distributed(args)

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

    model = training.setup_model(
        args.task, args.from_pretrained, args.model_config_file, args.model_type)
    optimizer = setup_optimizer(model, args.learning_rate)
    viz = utils.TBLogger(args.log_dir, exp_name, args.local_rank)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    if args.local_rank != -1:
        model = DDP(model)
    elif args.n_gpu > 1:
        model = nn.DataParallel(model)  # type: ignore

    train_dataset, train_loader = setup_dataset_and_loader(args, 'train')
    valid_dataset, valid_loader = setup_dataset_and_loader(args, 'valid')

    num_train_optimization_steps = get_num_train_optimization_steps(train_dataset, args)

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    trainer = training.Trainer(model, optimizer, scheduler, args)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num epochs = %d", args.num_train_epochs)
    logger.info("  Num train steps = %d", num_train_optimization_steps)

    for epoch_id in range(args.num_train_epochs):
        try:
            run_train_epoch(epoch_id, train_loader, trainer, viz, args)
            if not args.no_eval:
                run_valid_epoch(epoch_id, valid_loader, trainer, viz, args)
        except RuntimeError as e:
            if 'CUDA out of memory' in e.args[0]:
                eff_ngpu = utils.get_effective_num_gpus(args.local_rank, args.n_gpu)
                eff_batch_size = utils.get_effective_batch_size(
                    args.batch_size, args.local_rank, args.n_gpu,
                    args.gradient_accumulation_steps)
                message = (f"CUDA out of memory. Increase gradient_accumulation_steps to "
                           f"divide each batch over more forward passes.\n\n"
                           f"\tHyperparameters:\n"
                           f"\t\tbatch_size per backward-pass: {args.batch_size}\n"
                           f"\t\tgradient_accumulation_steps: "
                           f"{args.gradient_accumulation_steps}\n"
                           f"\t\tn_gpu: {eff_ngpu}\n"
                           f"\t\tbatch_size per (gpu * forward-pass): "
                           f"{eff_batch_size}")
                raise RuntimeError(message).with_traceback(e.__traceback__)
            raise

        # Save trained model
        if args.is_master and not (args.no_eval and epoch_id + 1 < args.num_train_epochs):
            logger.info("** ** * Saving trained model ** ** * ")
            # Only save the model itself
            output_model_dir = save_path / f"pytorch_model_{epoch_id}"
            output_model_dir.mkdir()
            model_to_save = getattr(model, 'module', model)
            model_to_save.save_pretrained(output_model_dir)
            logger.info(f"Saving model checkpoint to {output_model_dir}")


def run_eval(args: Optional[argparse.Namespace] = None) -> Dict[str, float]:
    if args is None:
        base_parser = create_base_parser()
        parser = create_eval_parser(base_parser)
        args = parser.parse_args()

    if args.from_pretrained is None:
        raise ValueError("Must specify pretrained model")
    if args.local_rank != -1:
        raise ValueError("TAPE does not support distributed validation pass")

    args = setup_distributed(args)
    utils.setup_logging(args.local_rank, save_path=None)
    utils.set_random_seeds(args.seed, args.n_gpu)

    pretrained_dir = Path(args.from_pretrained)

    logger.info(
        f"device: {args.device} "
        f"n_gpu: {args.n_gpu}")

    model = training.setup_model(
        args.task, args.from_pretrained, args.model_config_file, args.model_type)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)  # type: ignore

    valid_dataset, valid_loader = setup_dataset_and_loader(args, args.split)

    save_callbacks = [registry.get_callback(name) for name in args.save_callback]

    if len(args.metrics) > 0 and 'save_predictions' not in args.save_callback:
        save_callbacks.append(registry.get_callback('save_predictions'))
    metric_functions = [registry.get_metric(name) for name in args.metrics]

    save_outputs = run_eval_epoch(valid_loader, model, args, save_callbacks)

    target_key = getattr(model, 'module', model).TARGET_KEY
    prediction_key = getattr(model, 'module', model).PREDICTION_KEY
    metrics = {name: metric(save_outputs[target_key], save_outputs[prediction_key])
               for name, metric in zip(args.metrics, metric_functions)}
    save_outputs.update(metrics)
    logger.info(f'Evaluation Metrics: {metrics}')

    with (pretrained_dir / 'results.pkl').open('wb') as f:
        pkl.dump(save_outputs, f)

    return metrics


def run_embed(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        base_parser = create_base_parser()
        parser = create_embed_parser(base_parser)
        args = parser.parse_args()

    if args.from_pretrained is None:
        raise ValueError("Must specify pretrained model")
    if args.local_rank != -1:
        raise ValueError("TAPE does not support distributed embed pass")

    args = setup_distributed(args)
    utils.setup_logging(args.local_rank, save_path=None)
    utils.set_random_seeds(args.seed, args.n_gpu)

    logger.info(
        f"device: {args.device} "
        f"n_gpu: {args.n_gpu}")

    model = training.setup_model(
        args.model_type, args.task, args.from_pretrained, args.model_config_file)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)  # type: ignore

    dataset, loader = setup_dataset_and_loader(args, args.datafile)

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


def run_train_distributed(args: Optional[argparse.Namespace] = None) -> None:
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


def run_gridsearch(args: Optional[argparse.Namespace] = None, env=None) -> None:
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
            run_args.output_dir, run_args.exp_name, f'pytorch_model_{run_args.num_train_epochs - 1}')
        metrics = run_eval(copy(run_args))
        results.append((grid_args, metrics))
        shutil.rmtree(run_args.from_pretrained)

    for grid_args, metrics in results:
        print(grid_args)
        print(metrics)
        print()

    with open(os.path.join(args.output_dir, args.exp_name, 'gridsearch_results.pkl'), 'wb') as f:
        pkl.dump(results, f)


if __name__ == '__main__':
    # run_train()
    run_embed()
