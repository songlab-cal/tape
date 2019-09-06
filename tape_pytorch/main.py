from typing import Optional, Tuple, Union
import sys
import random
import math
from time import strftime, gmtime
from timeit import default_timer as timer
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers import (BertConfig, BertForMaskedLM, BertForPreTraining,
                                  AdamW, WarmupLinearSchedule)
from tensorboardX import SummaryWriter

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

from datasets import PfamTokenizer, PfamDataset, PfamBatch


logger = logging.getLogger(__name__)


class TBLogger:

    def __init__(self, log_dir: Union[str, Path], exp_name: str, local_rank: int):
        is_master = local_rank in (-1, 0)
        if is_master:
            log_dir = Path(log_dir) / exp_name
            logger.info(f"tensorboard file at: {log_dir}")
            self.logger = SummaryWriter(log_dir=str(log_dir))
        self._is_master = is_master

    def line_plot(self, step, val, split, key, xlabel="None") -> None:
        if self._is_master:
            self.logger.add_scalar(split + "/" + key, val, step)


@dataclass(frozen=False)
class TaskConfig:
    data_dir: str = 'data'
    vocab_file: str = 'data/pfam.model'
    pretrained_weight: Optional[str] = None
    log_dir: str = 'logs'
    output_dir: str = 'results'
    config_file: str = 'config/bert_config.json'
    # max_seq_length: Optional[int] = None
    train_batch_size: int = 512
    learning_rate: float = 1e-4
    num_train_epochs: int = 10
    warmup_steps: int = 10000
    cuda: bool = True
    on_memory: bool = False
    seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    loss_scale: int = 0
    num_workers: int = 20
    from_pretrained: bool = False
    exp_name: Optional[str] = None
    local_rank: int = -1
    bert_model: str = ''


class TaskRunner(object):

    def __init__(self, args: TaskConfig):
        super().__init__()
        is_master = args.local_rank in (-1, 0)
        device, n_gpu = self._setup_distributed(args.local_rank, args.cuda)

        if is_master:
            save_path, exp_name = self._get_savepath(args.output_dir, args.exp_name)
            save_path.mkdir(parents=True, exist_ok=False)
            # save all the hidden parameters.
            with (save_path / 'command.txt').open('w') as f:
                print(args, end='\n\n', file=f)

            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            save_files = Path(args.output_dir).iterdir()
            save_path = max(save_files, key=self._path_to_datetime)
            exp_name = save_path.name

        self._setup_logging(save_path, args.local_rank)

        logger.info(
            f"device: {device} "
            f"n_gpu: {n_gpu}, "
            f"distributed_training: {args.local_rank != -1}, "
            f"16-bits training: {args.fp16}")

        if args.gradient_accumulation_steps < 1:
            raise ValueError(
                f"Invalid gradient_accumulation_steps parameter: "
                f"{args.gradient_accumulation_steps}, should be >= 1")

        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

        self._set_random_seeds(args.seed, n_gpu)

        tokenizer = PfamTokenizer.from_pretrained(args.vocab_file)

        bert_config = BertConfig.from_json_file(args.config_file)

        model = self._setup_model(
            args.from_pretrained, args.bert_model, bert_config)  # , args.local_rank, n_gpu, args.fp16)

        optimizer = self._setup_optimizer(
            model, args.from_pretrained, args.fp16,
            args.learning_rate, args.pretrained_weight,
            args.loss_scale)

        if args.fp16:
            if not APEX_FOUND:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

        if args.local_rank != -1:
            if not APEX_FOUND:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model = DDP(model)
        elif n_gpu > 1:
            model = nn.DataParallel(model)

        # Store defaults
        self.exp_name = exp_name
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.device = device
        self.n_gpu = n_gpu
        self.max_grad_norm = 1.0

        # Store args
        self.data_dir = args.data_dir
        self.vocab_file = args.vocab_file
        self.pretrained_weight = args.pretrained_weight
        self.log_dir = args.log_dir
        self.output_dir = args.output_dir
        self.config_file = args.config_file
        self.train_batch_size = args.train_batch_size
        self.learning_rate = args.learning_rate
        self.num_train_epochs = args.num_train_epochs
        self.warmup_steps = args.warmup_steps
        self.cuda = args.cuda
        self.on_memory = args.on_memory
        self.seed = args.seed
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.fp16 = args.fp16
        self.loss_scale = args.loss_scale
        self.num_workers = args.num_workers
        self.from_pretrained = args.from_pretrained
        self.local_rank = args.local_rank
        self.bert_model = args.bert_model

    def _path_to_datetime(self, path: Path) -> datetime:
        name = path.name
        datetime_string = name.split('_')[0]
        year, month, day, time_string = datetime_string.split('-')
        hour, minute, second = time_string.split(':')
        pathdatetime = datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second))
        return pathdatetime

    def _setup_logging(self, save_path: Path, local_rank: int):
        log_level = logging.INFO if local_rank in (-1, 0) else logging.WARNING
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        file_handler = logging.FileHandler(save_path / 'log')
        file_handler.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%y/%m/%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    def _setup_model(self,
                     from_pretrained: bool,
                     bert_model: str,
                     config: BertConfig):

        if from_pretrained:
            model = BertForMaskedLM.from_pretrained(bert_model, config)
        else:
            model = BertForMaskedLM(config)

        model.cuda()

        return model

    def _setup_optimizer(self,
                         model: BertForPreTraining,
                         from_pretrained: bool,
                         fp16: bool,
                         learning_rate: float,
                         pretrained_weight: Optional[str],
                         loss_scale: int):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        if not from_pretrained:
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
        else:
            assert pretrained_weight is not None
            bert_weight_name = json.load(open("config/" + pretrained_weight + "_weight_name.json", "r"))
            optimizer_grouped_parameters = []
            for key, value in dict(model.named_parameters()).items():
                if value.requires_grad:
                    if key[12:] in bert_weight_name:
                        lr = learning_rate * 0.1
                    else:
                        lr = learning_rate

                    if any(nd in key for nd in no_decay):
                        optimizer_grouped_parameters += [
                            {"params": [value], "lr": lr, "weight_decay": 0.01}
                        ]

                    if not any(nd in key for nd in no_decay):
                        optimizer_grouped_parameters += [
                            {"params": [value], "lr": lr, "weight_decay": 0.0}
                        ]
        if from_pretrained:
            optimizer = AdamW(
                optimizer_grouped_parameters)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        return optimizer

    def _get_savepath(self, output_dir: str, exp_name: Optional[str]) -> Tuple[Path, str]:
        if exp_name is None:
            time_stamp = strftime("%y-%m-%d-%H:%M:%S", gmtime())
            exp_name = time_stamp + "_{:0>6d}".format(random.randint(0, int(1e6)))

        save_path = Path(output_dir) / exp_name
        return save_path, exp_name

    def _setup_distributed(self, local_rank: int, cuda: bool) -> Tuple[torch.device, int]:
        if local_rank != -1 and cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            n_gpu = 1
            torch.distributed.init_process_group(backend="nccl")
        elif not torch.cuda.is_available() or not cuda:
            device = torch.device("cpu")
            n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()

        return device, n_gpu

    def _set_random_seeds(self, seed: int, n_gpu: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def train(self):
        viz = TBLogger(self.log_dir, self.exp_name, self.local_rank)

        train_dataset = PfamDataset(self.data_dir, 'train', self.tokenizer)
        valid_dataset = PfamDataset(self.data_dir, 'valid', self.tokenizer)
        sampler_type = (DistributedSampler if self.local_rank != -1 else RandomSampler)

        train_loader = DataLoader(
            train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers,
            collate_fn=PfamBatch(), sampler=sampler_type(train_dataset))
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers,
            collate_fn=PfamBatch(), sampler=sampler_type(valid_dataset))

        num_train_optimization_steps = len(train_dataset)
        num_train_optimization_steps /= self.train_batch_size
        num_train_optimization_steps /= self.gradient_accumulation_steps
        num_train_optimization_steps *= self.num_train_epochs
        num_train_optimization_steps = int(num_train_optimization_steps)

        if self.local_rank != -1:
            num_train_optimization_steps //= torch.distributed.get_world_size()

        scheduler = WarmupLinearSchedule(
            self.optimizer, warmup_steps=self.warmup_steps, t_total=num_train_optimization_steps)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", self.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        self._iter_id = 0
        self._global_step = 0

        for epoch_id in range(self.num_train_epochs):
            self._run_train_epoch(
                epoch_id, train_loader, scheduler, viz)
            self._run_valid_epoch(
                epoch_id, valid_loader, viz)

            # Save trained model
            logger.info("** ** * Saving trained model ** ** * ")

            if self.local_rank in (-1, 0):
                # Only save the model itself
                output_model_dir = self.save_path / f"pytorch_model_{epoch_id}"
                self.model.save_pretrained(output_model_dir)

    def _run_train_epoch(self,
                         epoch_id: int,
                         train_loader: DataLoader,
                         scheduler: WarmupLinearSchedule,
                         viz: TBLogger):
        train_loss = 0
        num_train_examples = 0
        num_train_steps = 0
        loss_tmp = 0.

        torch.set_grad_enabled(True)
        self.model.train()

        start_t = timer()
        for step, batch in enumerate(train_loader):
            self._iter_id += 1
            batch = tuple(t.cuda(device=self.device, non_blocking=True) for t in batch)
            input_ids, input_mask, lm_label_ids, clan, family = batch
            outputs = self.model(
                input_ids, attention_mask=input_mask, masked_lm_labels=lm_label_ids)
            loss = outputs[0]

            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                # self.optimizer.backward(loss)
            else:
                loss.backward()

            if math.isnan(loss.item()):
                import pdb
                pdb.set_trace()

            train_loss += loss.item()

            viz.line_plot(self._iter_id, loss.item(), "loss", "train")

            loss_tmp += loss.item()

            num_train_examples += input_ids.size(0)
            num_train_steps += 1

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()
                self._global_step += 1

            if (step + 1) % 20 == 0:
                loss_tmp = loss_tmp / 20.0
                end_t = timer()
                time_stamp = strftime("%a %d %b %y %X", gmtime())

                ep = epoch_id + num_train_steps / float(len(train_loader))
                print_str = [
                    f"[{time_stamp}]",
                    f"[Ep: {ep:.2f}]",
                    f"[Iter: {num_train_steps}]",
                    f"[Time: {end_t - start_t:5.2f}s]",
                    f"[Loss: {loss_tmp:.5g}]",
                    f"[LR: {scheduler.get_lr()[0]:.5g}]"]
                start_t = end_t

                logger.info(''.join(print_str))
                loss_tmp = 0

    def _run_valid_epoch(self,
                         epoch_id: int,
                         valid_loader: DataLoader,
                         viz: TBLogger):
        num_batches = len(valid_loader)
        eval_loss = 0.

        torch.set_grad_enabled(False)
        self.model.eval()

        start_t = timer()
        for step, batch in enumerate(valid_loader):
            batch = tuple(t.cuda(device=self.device, non_blocking=True) for t in batch)
            input_ids, input_mask, lm_label_ids, clan, family = batch
            outputs = self.model(
                input_ids, attention_mask=input_mask, masked_lm_labels=lm_label_ids)
            loss = outputs[0]

            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu

            eval_loss += loss.item()

            end_t = timer()
            progress_string = f"\r Evaluating split val [{step + 1}/{num_batches}\t " \
                              f"Time: {end_t - start_t:5.2f}s]"

            logger.info(progress_string)

        eval_loss /= num_batches

        print_str = "Evaluation: [Loss: {eval_loss:.5g}]"

        logger.info(print_str)
        viz.line_plot(epoch_id, eval_loss, "loss", "val")


@click.command()
@click.option('--data-dir', default='data', type=click.Path(exists=True, file_okay=False))
@click.option('--vocab-file', default='data/pfam.model', type=click.Path(exists=True, dir_okay=False))
@click.option('--pretrained-weight', default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--log-dir', default='logs', type=click.Path())
@click.option('--output-dir', default='results', type=click.Path())
@click.option('--config-file', default='config/bert_config.json', type=click.Path(exists=True, dir_okay=False))
@click.option('--train-batch-size', default=512, type=int)
@click.option('--learning-rate', default=1e-4, type=float)
@click.option('--num-train-epochs', default=10, type=int)
@click.option('--warmup-steps', default=10000, type=int)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--on-memory', is_flag=True)
@click.option('--seed', default=42, type=int)
@click.option('--gradient-accumulation-steps', default=1, type=int)
@click.option('--fp16/--no-fp16', default=False)
@click.option('--loss-scale', default=0, type=int)
@click.option('--from-pretrained', is_flag=True)
@click.option('--exp-name', default=None, type=str)
@click.option('--local_rank', default=-1, type=int)
@click.option('--bert-model', default=str, type=str)
def main(data_dir: str = 'data',
         vocab_file: str = 'data/pfam.model',
         pretrained_weight: Optional[str] = None,
         log_dir: str = 'logs',
         output_dir: str = 'results',
         config_file: str = 'config/bert_config.json',
         # max_seq_length: Optional[int] = None,
         train_batch_size: int = 512,
         learning_rate: float = 1e-4,
         num_train_epochs: int = 10,
         warmup_steps: int = 10000,
         cuda: bool = True,
         on_memory: bool = False,
         seed: int = 42,
         gradient_accumulation_steps: int = 1,
         fp16: bool = False,
         loss_scale: int = 0,
         num_workers: int = 20,
         from_pretrained: bool = False,
         exp_name: Optional[str] = None,
         local_rank: int = -1,
         bert_model: str = ''):

    config = TaskConfig(
        data_dir, vocab_file, pretrained_weight, log_dir,
        output_dir, config_file, train_batch_size, learning_rate,
        num_train_epochs, warmup_steps, cuda,
        on_memory, seed, gradient_accumulation_steps,
        fp16, loss_scale, num_workers, from_pretrained,
        exp_name, local_rank, bert_model)

    runner = TaskRunner(config)
    runner.train()


if __name__ == '__main__':
    main()
