from typing import Optional, Tuple, Type
from time import strftime, gmtime
from timeit import default_timer as timer
import logging
from pathlib import Path
import json
from itertools import islice
from tqdm import tqdm

import click
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
from tape_pytorch.trainer import Trainer, RunConfig
import tape_pytorch.utils as utils
from tape_pytorch.datasets import TAPEDataset


logger = logging.getLogger(__name__)


def setup_model(model_type: str,
                task: str,
                from_pretrained: Optional[str],
                config_file: Optional[str]):

    model_cls = registry.get_task_model_class(task)
    if from_pretrained is not None:
        assert config_file is None
        load_dir = Path(from_pretrained)
        model = model_cls.from_pretrained(load_dir)
    else:
        if config_file is not None:
            config = TAPEConfig.from_json_file(config_file)
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


def setup_dataset_and_loader(run_config: RunConfig,
                             dataset_type: str) -> Tuple[TAPEDataset, DataLoader]:

    dataset_class: Type[TAPEDataset] = registry.get_dataset_class(  # type: ignore
        run_config.task)
    collate_fn_cls = registry.get_collate_fn_class(run_config.task)
    sampler_type = (DistributedSampler if run_config.local_rank != -1 else RandomSampler)

    dataset = dataset_class(run_config.data_dir, dataset_type, run_config.tokenizer)

    loader = DataLoader(  # type: ignore
        dataset,
        batch_size=run_config.batch_size_per_gpu_forward,  # actual batch size
        num_workers=run_config.num_workers,
        collate_fn=collate_fn_cls(),
        sampler=sampler_type(dataset))

    return loader


def get_num_train_optimization_steps(train_loader: DataLoader, run_config: RunConfig) -> int:
    return int(len(train_loader) / run_config.train_batch_size * run_config.num_train_epochs)


def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    trainer: Trainer,
                    viz: utils.TBLogger,
                    run_config: RunConfig):
    metrics = utils.MetricsAccumulator(smoothing=1 - 1 / run_config.num_log_iter)

    torch.set_grad_enabled(True)
    trainer.model.train()

    start_t = timer()

    if run_config.debug:
        train_loader = islice(train_loader, 10)  # type: ignore

    loss_tmp = 0.
    for step, batch in enumerate(train_loader):
        loss = trainer.forward(batch)
        loss /= run_config.gradient_accumulation_steps
        loss_tmp += loss.item()
        trainer.backward(loss)

        if (step + 1) % run_config.gradient_accumulation_steps == 0:
            trainer.step()
            metrics.update(loss_tmp)
            viz.line_plot(trainer.global_step, loss_tmp, "loss", "train")
            loss_tmp = 0.

        if (step + 1) % run_config.num_log_iter == 0:
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


def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    trainer: Trainer,
                    viz: utils.TBLogger,
                    run_config: RunConfig):
    num_batches = len(valid_loader)
    eval_loss = 0.

    torch.set_grad_enabled(False)
    trainer.model.eval()

    if run_config.debug:
        valid_loader = islice(valid_loader, 10)  # type: ignore

    for step, batch in tqdm(enumerate(valid_loader), desc='Evaluating split val'):
        loss = trainer.forward(batch)
        eval_loss += loss.item()

    eval_loss /= num_batches

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"

    logger.info(print_str)
    viz.line_plot(epoch_id, eval_loss, "loss", "val")


@click.command()
@click.argument('task', type=click.Choice(registry.dataset_name_mapping.keys()))
@click.argument('model_type',
                type=click.Choice(registry.model_name_mapping.keys()))
@click.option('--config-file', default=None,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--data-dir', default='data', type=click.Path(exists=True, file_okay=False))
@click.option('--vocab-file', default='data/pfam.model',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--pretrained-weight', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--log-dir', default='logs', type=click.Path())
@click.option('--output-dir', default='results', type=click.Path())
@click.option('--train-batch-size', default=1024, type=int)
@click.option('--learning-rate', default=1e-4, type=float)
@click.option('--num-train-epochs', default=10, type=int)
@click.option('--num-log-iter', default=20, type=int)
@click.option('--warmup-steps', default=10000, type=int)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--on-memory', is_flag=True)
@click.option('--seed', default=42, type=int)
@click.option('--gradient-accumulation-steps', default=1, type=int)
@click.option('--fp16/--no-fp16', default=False)
@click.option('--loss-scale', default=0, type=int)
@click.option('--from-pretrained', default=None,
              type=click.Path(exists=True, file_okay=False))
@click.option('--exp-name', default=None, type=str)
@click.option('--local_rank', default=-1, type=int)
@click.option('--tokenizer', type=click.Choice(['bpe', 'dummy']), default='bpe')
@click.option('--max-grad-norm', default=1.0, type=float)
@click.option('--debug/--no-debug', default=False)
def main(task: str,
         model_type: str,
         config_file: Optional[str] = None,
         data_dir: str = 'data',
         vocab_file: str = 'data/pfam.model',
         pretrained_weight: Optional[str] = None,
         log_dir: str = 'logs',
         output_dir: str = 'results',
         train_batch_size: int = 1024,
         learning_rate: float = 1e-4,
         num_train_epochs: int = 10,
         num_log_iter: int = 20,
         warmup_steps: int = 10000,
         cuda: bool = True,
         on_memory: bool = False,
         seed: int = 42,
         gradient_accumulation_steps: int = 1,
         fp16: bool = False,
         loss_scale: int = 0,
         num_workers: int = 20,
         from_pretrained: Optional[str] = None,
         exp_name: Optional[str] = None,
         local_rank: int = -1,
         tokenizer: str = 'bpe',
         max_grad_norm: float = 1.0,
         debug: bool = False):

    run_config = RunConfig(
        local_rank, task, data_dir, num_train_epochs, num_log_iter, tokenizer, cuda, fp16,
        max_grad_norm, gradient_accumulation_steps, train_batch_size, seed, num_workers,
        warmup_steps, debug)

    save_path, exp_name = utils.get_savepath_and_expname(
        output_dir, exp_name, run_config.is_master)

    if run_config.is_master:
        # save all the hidden parameters.
        with (save_path / 'command.txt').open('w') as f:
            json.dump(run_config.to_dict(), f)

    utils.setup_logging(save_path, run_config.local_rank)
    utils.set_random_seeds(run_config.seed, run_config.n_gpu)

    logger.info(
        f"device: {run_config.device} "
        f"n_gpu: {run_config.n_gpu}, "
        f"distributed_training: {run_config.local_rank != -1}, "
        f"16-bits training: {run_config.use_fp16}")

    tokenizer = registry.get_tokenizer_class(tokenizer).from_pretrained(vocab_file)
    model = setup_model(model_type, task, from_pretrained, config_file)
    optimizer = setup_optimizer(model, learning_rate)
    viz = utils.TBLogger(log_dir, exp_name, run_config.local_rank)

    if run_config.use_fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    if run_config.local_rank != -1:
        model = DDP(model)
    elif run_config.n_gpu > 1:
        model = nn.DataParallel(model)

    train_dataset, train_loader = setup_dataset_and_loader(run_config, 'train')
    valid_dataset, valid_loader = setup_dataset_and_loader(run_config, 'valid')

    num_train_optimization_steps = get_num_train_optimization_steps(train_loader, run_config)

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=run_config.warmup_steps, t_total=num_train_optimization_steps)

    trainer = Trainer(model, optimizer, scheduler, run_config)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", run_config.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    for epoch_id in range(num_train_epochs):
        run_train_epoch(epoch_id, train_loader, trainer, viz, run_config)
        run_valid_epoch(epoch_id, valid_loader, trainer, viz, run_config)

        # Save trained model
        logger.info("** ** * Saving trained model ** ** * ")

        if run_config.is_master:
            # Only save the model itself
            output_model_dir = save_path / f"pytorch_model_{epoch_id}"
            output_model_dir.mkdir()
            model_to_save = getattr(model, 'module', model)
            model_to_save.save_pretrained(output_model_dir)
            logger.info(f"Saving model checkpoint to {output_model_dir}")


if __name__ == '__main__':
    main()
