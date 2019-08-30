from typing import Optional, Tuple
import os
import random
from time import strftime, gmtime
import logging
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import BertConfig, BertForMaskedLM, BertForPreTraining, AdamW
from configfactory import Registry

config = Registry()


logger = logging.getLogger(__name__)


class TaskRunner(object):

    def __init__(self, args):
        super().__init__()

        save_path = self._get_savepath(args.output_dir, args.save_name)
        save_path.mkdir(parents=True, exist_ok=True)

        # save all the hidden parameters.
        with (save_path / 'command.txt').open('w') as f:
            print(args, end='\n\n', file=f)

        device, n_gpu = self._setup_distributed(args.local_rank, args.no_cuda)

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

        tokenizer = None  # TODO: Make a tokenizer

        config = BertConfig.from_json_file(args.config_file)

        model = self._setup_model(
            args.from_pretrained, args.bert_model, config, args.local_rank, n_gpu, args.fp16)

    def _setup_model(self,
                     from_pretrained: bool,
                     bert_model: str,
                     config: BertConfig,
                     local_rank: int,
                     n_gpu: int,
                     fp16: bool) -> BertForPreTraining:

        if from_pretrained:
            model = BertForMaskedLM.from_pretrained(bert_model, config)
        else:
            model = BertForMaskedLM(config)

        if fp16:
            model.half()

        if local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex "
                    "to use distributed and fp16 training.")
            model = DDP(model)
        elif n_gpu > 1:
            model = nn.DataParallel(model)

        model.cuda()
        return model

    def _setup_optimizer(self,
                         model: BertForPreTraining,
                         from_pretrained: bool,
                         fp16: bool,
                         learning_rate: float,
                         pretrained_weight: str,
                         loss_scale: int,
                         warmup_steps: int,
                         num_train_optimization_steps: int):
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
        # set different parameters for vision branch and lanugage branch.
        if fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                )

            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                bias_correction=False,
                max_grad_norm=1.0,
            )
            if loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)

        else:
            if from_pretrained:
                optimizer = AdamW(
                    optimizer_grouped_parameters)
            else:
                optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        return optimizer

    def _get_savepath(self, output_dir: str, save_name: Optional[str]) -> Path:
        if save_name is None:
            time_stamp = strftime("%d-%b-%y-%X-%a", gmtime())
            save_name = time_stamp + "_{:0>6d}".format(random.randint(0, int(1e6)))

        save_path = Path(output_dir) / save_name

        return save_path

    def _setup_distributed(self, local_rank: int, no_cuda: bool) -> Tuple[torch.device, int]:
        if local_rank != -1 and not no_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            n_gpu = 1
            torch.distributed.init_process_group(backend="nccl")
        elif not torch.cuda.is_available() or no_cuda:
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
        viz = TBlogger("logs", timeStamp)

        train_dataset = PfamLoaderTrain(
            args.train_file,
            tokenizer,
            seq_len=args.max_seq_length,
            batch_size=args.train_batch_size,
            predict_feature=args.predict_feature,
            num_workers=args.num_workers)

        validation_dataset = PfamLoaderVal(
            args.validation_file,
            tokenizer,
            seq_len=args.max_seq_length,
            batch_size=args.train_batch_size,
            predict_feature=args.predict_feature,
            num_workers=args.num_workers,
        )

        num_train_optimization_steps = len(train_dataset)
        num_train_optimization_steps /= self.train_batch_size
        num_train_optimization_steps /= self.gradient_accumulation_steps
        num_train_optimization_steps *= self.num_train_epochs
        num_train_optimization_steps = int(num_train_optimization_steps)

        if self.local_rank != -1:
            num_train_optimization_steps //= torch.distributed.get_world_size()


@config.register
def main(train_file: str,
         validation_file: str,
         pretrained_weight: Optional[str] = None,
         output_dir: str = 'results',
         config_file: str = 'config/bert_config.json',
         max_seq_length: Optional[int] = None,
         train_batch_size: int = 512,
         learning_rate: float = 1e-4,
         num_train_epochs: int = 10,
         warmup_steps: int = 10000,
         no_cuda: bool = False,
         on_memory: bool = False,
         seed: int = 42,
         gradient_accumulation_steps: int = 1,
         fp16: bool = False,
         loss_scale: float = 0,
         num_workers: int = 20,
         from_pretrained: bool = False,
         save_name: Optional[str] = None):

    parser = config.get_parser()
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()

    runner = TaskRunner(args)
