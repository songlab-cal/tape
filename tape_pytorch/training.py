import typing
import logging
from timeit import default_timer as timer
import itertools
from collections import ChainMap
import json

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_transformers import WarmupLinearSchedule

import tape_pytorch.utils as utils

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
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
                    gradient_accumulation_steps: int = 1) -> float:
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
                ep = epoch_id + step / float(len(train_loader))
                print_str = [
                    f"[Ep: {ep:.2f}]",
                    f"[Iter: {runner.global_step}]",
                    f"[Time: {end_t - start_t:5.2f}s]",
                    f"[Loss: {metrics.loss():.5g}]",
                    f"[LR: {runner.scheduler.get_lr()[0]:.5g}]"]  # type: ignore
                start_t = end_t

                logger.info(''.join(print_str))
    logger.info(f"Train: [Loss: {metrics.final_loss():.5g}]")
    return metrics.final_loss()


def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    runner: ForwardRunner,
                    viz: utils.TBLogger,
                    is_master: bool = True) -> float:
    num_batches = len(valid_loader)
    eval_loss = 0.

    torch.set_grad_enabled(False)
    runner.model.eval()

    for batch in tqdm(valid_loader, desc='Running Eval', total=num_batches,
                      disable=not is_master, leave=False):
        loss: LossType = runner.forward(batch)  # type: ignore
        eval_loss += loss.item()

    eval_loss /= num_batches

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"

    logger.info(print_str)
    viz.line_plot(epoch_id, eval_loss, "loss", "val")

    return eval_loss


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
    eval_loss = utils.reduce_scalar(eval_loss)
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


def run_train(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_log_iter: int = 20,
              fp16: bool = False,
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              log_dir: str = './logs',
              no_eval: bool = False,
              save_freq: typing.Union[int, str] = 1,
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './data',
              vocab_file: str = './data/pfam.model',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'bpe',
              num_workers: int = 16,
              debug: bool = False,
              patience: int = -1) -> None:
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    save_path, exp_name = utils.get_savepath_and_expname(
        output_dir, exp_name, is_master)

    if is_master:
        # save all the hidden parameters.
        with (save_path / 'config.json').open('w') as f:
            json.dump(input_args, f)

    utils.setup_logging(local_rank, save_path)
    utils.set_random_seeds(seed, n_gpu)

    model = utils.setup_model(
        task, from_pretrained, model_config_file, model_type)
    optimizer = utils.setup_optimizer(model, learning_rate)
    viz = utils.TBLogger(log_dir, exp_name, local_rank)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    if fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    if local_rank != -1:
        model = DDP(model)
    elif n_gpu > 1:
        model = nn.DataParallel(model)  # type: ignore

    train_dataset = utils.setup_dataset(task, data_dir, 'train', tokenizer)
    valid_dataset = utils.setup_dataset(task, data_dir, 'valid', tokenizer)
    train_loader = utils.setup_loader(
        task, train_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)
    valid_loader = utils.setup_loader(
        task, valid_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    runner = BackwardRunner(
        model, optimizer, scheduler, device, n_gpu, fp16, max_grad_norm)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)
    is_master = local_rank in (-1, 0)

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and no_eval:
        raise ValueError("Cannot set save_freq to 'improvement' and no_eval")

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num epochs = %d", num_train_epochs)
    logger.info("  Num train steps = %d", num_train_optimization_steps)

    best_val_loss = float('inf')
    num_epochs_no_improvement = 0

    def do_save(epoch_id: int, num_epochs_no_improvement: int) -> bool:
        if not is_master:
            return False
        if isinstance(save_freq, int):
            return ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs)
        else:
            return num_epochs_no_improvement == 0

    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        for epoch_id in range(num_train_epochs):
            run_train_epoch(epoch_id, train_loader, runner,
                            viz, num_log_iter, gradient_accumulation_steps)
            if not no_eval:
                val_loss = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    num_epochs_no_improvement = 0
                else:
                    num_epochs_no_improvement += 1
                # Reduce loss across all processes if multiprocessing
                val_loss = utils.reduce_scalar(val_loss)

            # Save trained model
            if do_save(epoch_id, num_epochs_no_improvement):
                logger.info("** ** * Saving trained model ** ** * ")
                # Only save the model itself
                output_model_dir = save_path / f"pytorch_model_{epoch_id}"
                output_model_dir.mkdir()
                model_to_save = getattr(model, 'module', model)
                model_to_save.save_pretrained(output_model_dir)
                logger.info(f"Saving model checkpoint to {output_model_dir}")

            if patience > 0 and num_epochs_no_improvement >= patience:
                logger.info(f"Finished training at epoch {epoch_id} because no "
                            f"improvement for {num_epochs_no_improvement} epochs.")
                utils.barrier_if_distributed()
                break
    logger.info(f"Finished training after {num_train_epochs} epochs.")

    if not no_eval:
        logger.info(f"Best Val Loss: {best_val_loss}")
    utils.barrier_if_distributed()
