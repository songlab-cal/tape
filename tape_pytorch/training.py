import typing
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers import AdamW
from pytorch_transformers.modeling_utils import PreTrainedModel

import tape_pytorch.models as models
import tape_pytorch.utils as utils
from tape_pytorch.registry import registry
from tape_pytorch.datasets import TAPEDataset

try:
    from apex import amp
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

logger = logging.getLogger(__name__)


def setup_model(task: str,
                from_pretrained: typing.Optional[str] = None,
                model_config_file: typing.Optional[str] = None,
                model_type: typing.Optional[str] = None) -> models.TAPEPreTrainedModel:

    if from_pretrained is not None:
        model = models.from_pretrained(task, from_pretrained)
    elif model_config_file is not None:
        model = models.from_config(task, model_config_file)
    elif model_type is not None:
        model = models.from_model_type(task, model_type)
    else:
        raise ValueError(
            "Must specify one of <from_pretrained, model_config_file, or model_type>")
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


def setup_dataset(task: str,
                  data_dir: typing.Union[str, Path],
                  split: str,
                  tokenizer: str) -> TAPEDataset:
    dataset_class: typing.Type[TAPEDataset] = registry.get_dataset_class(  # type: ignore
        task)
    dataset = dataset_class(data_dir, split, tokenizer)
    return dataset


def setup_loader(task: str,
                 dataset: TAPEDataset,
                 batch_size: int,
                 local_rank: int,
                 n_gpu: int,
                 gradient_accumulation_steps: int,
                 num_workers: int) -> DataLoader:
    collate_fn_cls = registry.get_collate_fn_class(task)
    sampler_type = (DistributedSampler if local_rank != -1 else RandomSampler)

    loader = DataLoader(  # type: ignore
        dataset,
        batch_size=utils.get_effective_batch_size(
            batch_size, local_rank, n_gpu, gradient_accumulation_steps),
        num_workers=num_workers,
        collate_fn=collate_fn_cls(),
        sampler=sampler_type(dataset))

    return loader


class ForwardRunner:

    def __init__(self,
                 model: nn.Module,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1):

        self.model = model
        self.device = device
        self.n_gpu = n_gpu
        self._loss_key = getattr(model, 'module', model).LOSS_KEY

    def forward(self, batch: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.device.type == 'cuda':
            batch = {name: tensor.cuda(device=self.device, non_blocking=True)
                     for name, tensor in batch.items()}
        outputs = self.model(**batch)
        loss = outputs[self.loss_key]

        if self.n_gpu > 1:
            loss = loss.mean()

        return loss

    @property
    def loss_key(self) -> str:
        return self._loss_key


class BackwardRunner(ForwardRunner):

    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,  # type: ignore
                 scheduler: optim.lr_scheduler.LambdaLR,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 max_grad_norm: float = 1.0):

        super().__init__(model, device, n_gpu)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fp16 = fp16
        self.max_grad_norm = max_grad_norm
        self._global_step = 0

    def backward(self, loss) -> None:
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()  # type: ignore
        self.optimizer.zero_grad()
        self._global_step += 1

    @property
    def global_step(self) -> int:
        return self._global_step
