from typing import Optional, Tuple, Type, Callable, Sequence, List, Dict, Any
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
from tape_pytorch.models.task_models import TAPEConfig
from tape_pytorch.trainer import Trainer
import tape_pytorch.utils as utils
from tape_pytorch.datasets import TAPEDataset


logger = logging.getLogger(__name__)
warnings.filterwarnings(  # Ignore pytorch warning about loss gathering
    'ignore', message='Was asked to gather along dimension 0', module='torch.nn.parallel')


def setup_model(model_type: str,
                task: str,
                from_pretrained: Optional[str],
                model_config_file: Optional[str]):

    model_cls = registry.get_task_model_class(task)
    if from_pretrained is not None:
        assert model_config_file is None
        load_dir = Path(from_pretrained)
        model = model_cls.from_pretrained(load_dir)
    else:
        if model_config_file is not None:
            config = TAPEConfig.from_json_file(model_config_file)
        else:
            base_config = registry.get_model_class(model_type).config_class()
            config = TAPEConfig(base_config, base_model=model_type)
        model = model_cls(config)

    model.cuda()

    return model


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
        torch.distributed.init_process_group(backend="nccl")
    elif not torch.cuda.is_available() or args.no_cuda:
        args.device = torch.device("cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()

    args.is_master = args.local_rank in (-1, 0)

    return args


def get_effective_num_gpus(args: argparse.Namespace) -> int:
    if args.local_rank == -1:
        num_gpus = args.n_gpu
    else:
        num_gpus = torch.distributed.get_world_size()
    return num_gpus


def get_effective_batch_size(args: argparse.Namespace) -> int:
    batch_size = float(args.batch_size)
    batch_size /= getattr(args, 'gradient_accumulation_steps', 1)
    batch_size /= get_effective_num_gpus(args)
    return int(batch_size)


def setup_dataset_and_loader(args: argparse.Namespace,
                             dataset_type: str) -> Tuple[TAPEDataset, DataLoader]:

    dataset_class: Type[TAPEDataset] = registry.get_dataset_class(  # type: ignore
        args.task)
    collate_fn_cls = registry.get_collate_fn_class(args.task)
    sampler_type = (DistributedSampler if args.local_rank != -1 else RandomSampler)

    dataset = dataset_class(args.data_dir, dataset_type, args.tokenizer)

    loader = DataLoader(  # type: ignore
        dataset,
        batch_size=get_effective_batch_size(args),
        num_workers=args.num_workers,
        collate_fn=collate_fn_cls(),
        sampler=sampler_type(dataset))

    return dataset, loader


def get_num_train_optimization_steps(train_dataset: TAPEDataset,
                                     args: argparse.Namespace) -> int:
    return int(len(train_dataset) / args.batch_size * args.num_train_epochs)


def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    trainer: Trainer,
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
                    trainer: Trainer,
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
    parser.add_argument('task', choices=list(registry.dataset_name_mapping.keys()),
                        help='TAPE Task to train/eval on')
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
    return parser


def create_eval_parser(base_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run Eval on the TAPE Datasets',
                                     parents=[base_parser])
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
    return parser


def run_train():
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

    model = setup_model(
        args.model_type, args.task, args.from_pretrained, args.model_config_file)
    optimizer = setup_optimizer(model, args.learning_rate)
    viz = utils.TBLogger(args.log_dir, exp_name, args.local_rank)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    if args.local_rank != -1:
        model = DDP(model)
    elif args.n_gpu > 1:
        model = nn.DataParallel(model)

    train_dataset, train_loader = setup_dataset_and_loader(args, 'train')
    valid_dataset, valid_loader = setup_dataset_and_loader(args, 'valid')

    num_train_optimization_steps = get_num_train_optimization_steps(train_dataset, args)

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    trainer = Trainer(model, optimizer, scheduler, args)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num epochs = %d", args.num_train_epochs)
    logger.info("  Num train steps = %d", num_train_optimization_steps)

    for epoch_id in range(args.num_train_epochs):
        try:
            run_train_epoch(epoch_id, train_loader, trainer, viz, args)
            run_valid_epoch(epoch_id, valid_loader, trainer, viz, args)
        except RuntimeError as e:
            if 'CUDA out of memory' in e.args[0]:
                message = (f"CUDA out of memory. Increase gradient_accumulation_steps to "
                           f"divide each batch over more forward passes.\n\n"
                           f"\tHyperparameters:\n"
                           f"\t\tbatch_size per backward-pass: {args.batch_size}\n"
                           f"\t\tgradient_accumulation_steps: "
                           f"{args.gradient_accumulation_steps}\n"
                           f"\t\tn_gpu: {get_effective_num_gpus(args)}\n"
                           f"\t\tbatch_size per (gpu * forward-pass): "
                           f"{get_effective_batch_size(args)}")
                raise RuntimeError(message).with_traceback(e.__traceback__)
            raise

        # Save trained model
        logger.info("** ** * Saving trained model ** ** * ")

        if args.is_master:
            # Only save the model itself
            output_model_dir = save_path / f"pytorch_model_{epoch_id}"
            output_model_dir.mkdir()
            model_to_save = getattr(model, 'module', model)
            model_to_save.save_pretrained(output_model_dir)
            logger.info(f"Saving model checkpoint to {output_model_dir}")


def run_eval():
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

    model = setup_model(
        args.model_type, args.task, args.from_pretrained, args.model_config_file)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    valid_dataset, valid_loader = setup_dataset_and_loader(args, 'valid')

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


if __name__ == '__main__':
    run_train()
