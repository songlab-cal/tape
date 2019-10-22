import typing
import logging
from time import strftime, gmtime
from timeit import default_timer as timer
import itertools
from collections import ChainMap

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tape_pytorch.utils as utils

try:
    from apex import amp
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

logger = logging.getLogger(__name__)

LossType = torch.Tensor
OutputDict = typing.Dict[str, typing.Any]
ForwardModelOutput = typing.Union[LossType, typing.Tuple[LossType, OutputDict]]


class ForwardRunner:

    def __init__(self,
                 model: nn.Module,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1):

        self.model = model
        self.device = device
        self.n_gpu = n_gpu
        self._loss_key = getattr(model, 'module', model).LOSS_KEY

    def forward(self,
                batch: typing.Dict[str, torch.Tensor],
                return_outputs: bool = False) -> ForwardModelOutput:
        if self.device.type == 'cuda':
            batch = {name: tensor.cuda(device=self.device, non_blocking=True)
                     for name, tensor in batch.items()}
        outputs = self.model(**batch)
        loss = outputs[self.loss_key]

        if self.n_gpu > 1:
            loss = loss.mean()

        if return_outputs:
            return loss, outputs
        else:
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


def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    runner: BackwardRunner,
                    viz: utils.TBLogger,
                    num_log_iter: int = 20,
                    gradient_accumulation_steps: int = 1) -> None:
    metrics = utils.MetricsAccumulator(smoothing=1 - 1 / num_log_iter)

    torch.set_grad_enabled(True)
    runner.model.train()

    start_t = timer()

    loss_tmp = 0.
    for step, batch in enumerate(train_loader):
        loss: LossType = runner.forward(batch)  # type: ignore
        loss /= gradient_accumulation_steps
        loss_tmp += loss.item()
        runner.backward(loss)

        if (step + 1) % gradient_accumulation_steps == 0:
            runner.step()
            metrics.update(loss_tmp)
            viz.line_plot(runner.global_step, loss_tmp, "loss", "train")
            loss_tmp = 0.

            if runner.global_step % num_log_iter == 0:
                end_t = timer()
                time_stamp = strftime("%y-%m-%d %X", gmtime())
                ep = epoch_id + step / float(len(train_loader))
                print_str = [
                    f"[{time_stamp}]",
                    f"[Ep: {ep:.2f}]",
                    f"[Iter: {runner.global_step}]",
                    f"[Time: {end_t - start_t:5.2f}s]",
                    f"[Loss: {metrics.loss():.5g}]",
                    f"[LR: {runner.scheduler.get_lr()[0]:.5g}]"]  # type: ignore
                start_t = end_t

                logger.info(''.join(print_str))

    logger.info(f"Train: [Loss: {metrics.final_loss():.5g}]")


def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    runner: ForwardRunner,
                    viz: utils.TBLogger,
                    is_master: bool = True) -> None:
    num_batches = len(valid_loader)
    eval_loss = 0.

    torch.set_grad_enabled(False)
    runner.model.eval()

    for batch in tqdm(valid_loader, desc='Evaluating split val', total=num_batches,
                      disable=not is_master):
        loss: LossType = runner.forward(batch)  # type: ignore
        eval_loss += loss.item()

    eval_loss /= num_batches

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"

    logger.info(print_str)
    viz.line_plot(epoch_id, eval_loss, "loss", "val")


def run_eval_epoch(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   is_master: bool = True,
                   save_callback: typing.Optional[typing.Sequence[typing.Callable]] = None) \
        -> typing.Dict[str, typing.List[typing.Any]]:
    num_batches = len(eval_loader)
    eval_loss = 0.
    torch.set_grad_enabled(False)
    runner.model.eval()

    save_outputs = []

    for batch in tqdm(eval_loader, desc='Evaluating split val', total=num_batches,
                      disable=not is_master):
        loss, outputs = runner.forward(batch, return_outputs=True)  # type: ignore
        if save_callback is not None:
            to_save = dict(ChainMap(
                *(callback(runner.model, batch, outputs) for callback in save_callback)))
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
