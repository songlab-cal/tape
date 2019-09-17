from typing import Dict
import logging
import argparse

import torch
import torch.nn as nn

try:
    from apex import amp
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 args: argparse.Namespace):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self._global_step = 0
        self._loss_key = getattr(model, 'module', model).LOSS_KEY

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cuda_batch = {name: tensor.cuda(device=self.args.device, non_blocking=True)
                      for name, tensor in batch.items()}
        outputs = self.model(**cuda_batch)
        loss = outputs[self._loss_key]

        if self.args.n_gpu > 1:
            loss = loss.mean()

        return loss

    def backward(self, loss) -> None:
        if self.args.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()  # type: ignore
        self.optimizer.zero_grad()
        self._global_step += 1

    @property
    def global_step(self) -> int:
        return self._global_step
