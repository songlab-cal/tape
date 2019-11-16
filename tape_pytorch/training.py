import typing
import os
import logging
from timeit import default_timer as timer
import itertools
from collections import ChainMap
import json
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_transformers import WarmupLinearSchedule

from protein_models import ProteinModel
import tape_pytorch.utils as utils
import tape_pytorch.errors as errors
import tape_pytorch.models as models
import tape_pytorch.visualization as visualization

try:
    from apex import amp
    import amp_C
    import apex_C
    from apex.amp import _amp_state
    from apex.parallel.distributed import flat_dist_call
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

logger = logging.getLogger(__name__)

MetricsDict = typing.Dict[str, float]
LossAndMetrics = typing.Tuple[float, MetricsDict]
OutputDict = typing.Dict[str, typing.Any]
ForwardModelOutput = typing.Union[typing.Tuple[torch.Tensor, OutputDict],
                                  typing.Tuple[torch.Tensor, OutputDict, OutputDict]]


class ForwardRunner:

    def __init__(self,
                 model: ProteinModel,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1):

        self.model = model
        self.device = device
        self.n_gpu = n_gpu
        model = getattr(model, 'module', model)

    def forward(self,
                batch: typing.Dict[str, torch.Tensor],
                return_outputs: bool = False) -> ForwardModelOutput:
        if self.device.type == 'cuda':
            batch = {name: tensor.cuda(device=self.device, non_blocking=True)
                     for name, tensor in batch.items()}
        outputs = self.model(**batch)
        # loss = outputs[self.loss_key]
        # metrics = outputs[self.metrics_key]
        loss = outputs[0]
        metrics: typing.Dict[str, torch.Tensor] = {}

        if self.n_gpu > 1:
            loss = loss.mean()
            metrics = {name: metric.mean() for name, metric in metrics.items()}

        if return_outputs:
            return loss, metrics, outputs
        else:
            return loss, metrics


class BackwardRunner(ForwardRunner):

    def __init__(self,
                 model: ProteinModel,
                 optimizer: optim.Optimizer,  # type: ignore
                 scheduler: typing.Optional[optim.lr_scheduler.LambdaLR] = None,
                 gradient_accumulation_steps: int = 1,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 max_grad_norm: float = 1.0,
                 local_rank=-1):

        super().__init__(model, device, n_gpu)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fp16 = fp16
        self.max_grad_norm = max_grad_norm
        self._global_step = 0
        self._local_rank = local_rank
        self._overflow_buf = torch.cuda.IntTensor([0])  # type: ignore
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def backward(self, loss) -> None:
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer, delay_overflow_check=True) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        if self._local_rank == -1 or not self.fp16:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self._step()
        else:
            self._step_distributed_fp16()

    def _step(self) -> None:
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()  # type: ignore
        self._global_step += 1

    def _step_distributed_fp16(self) -> None:
        # Optimized step function from https://github.com/NVIDIA/DeepLearningExamples
        # Only performs allreduce after gradient accumulation, which reduces communication
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(self.optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        # allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else \
            # torch.float32
        allreduce_dtype = torch.float16
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / (
                torch.distributed.get_world_size() * self.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [allreduced_views, master_grads],
            1. / scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = self._overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            self._step()
        else:
            # Overflow detected, print message and clear gradients
            logger.info(f"Gradient overflow.  Skipping step, reducing loss scale to "
                        f"{scaler.loss_scale()}")
            if _amp_state.opt_properties.master_weights:
                for param in self.optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in self.model.parameters():
            param.grad = None

    @property
    def global_step(self) -> int:
        return self._global_step


def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    runner: BackwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    num_log_iter: int = 20,
                    gradient_accumulation_steps: int = 1) -> LossAndMetrics:
    if viz is None:
        viz = visualization.DummyVisualizer()
    smoothing = 1 - 1 / num_log_iter
    accumulator = utils.MetricsAccumulator(smoothing)

    torch.set_grad_enabled(True)
    runner.model.train()

    start_t = timer()
    for step, batch in enumerate(train_loader):
        loss, metrics = runner.forward(batch)  # type: ignore
        runner.backward(loss)
        accumulator.update(
            loss.item(), {name: value.item() for name, value in metrics.items()}, step=False)
        if (step + 1) % gradient_accumulation_steps == 0:
            runner.step()
            loss_tmp, metrics_tmp = accumulator.step()
            metrics_tmp['loss'] = loss_tmp
            viz.log_metrics(metrics_tmp, "train", runner.global_step)
            if runner.global_step % num_log_iter == 0:
                end_t = timer()
                ep = epoch_id + step / float(len(train_loader))
                print_str = [
                    f"[Ep: {ep:.2f}]",
                    f"[Iter: {runner.global_step}]",
                    f"[Time: {end_t - start_t:5.2f}s]",
                    f"[Loss: {accumulator.loss():.5g}]"]
                print_str += [f"[{name.capitalize()}: {value:.5g}]"
                              for name, value in accumulator.metrics().items()]
                if runner.scheduler is not None:
                    curr_lr = runner.scheduler.get_lr()[0]  # type: ignore
                else:
                    curr_lr = runner.optimizer.param_groups[0]['lr']
                print_str += [f"[LR: {curr_lr:.5g}]"]
                start_t = end_t

                logger.info(''.join(print_str))

    final_print_str = f"Train: [Loss: {accumulator.final_loss():.5g}]"
    for name, value in accumulator.final_metrics().items():
        final_print_str += f"[{name.capitalize()}: {value:.5g}]"
    logger.info(final_print_str)
    return accumulator.final_loss(), accumulator.final_metrics()


def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    runner: ForwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    is_master: bool = True) -> typing.Tuple[float, typing.Dict[str, float]]:
    num_batches = len(valid_loader)
    accumulator = utils.MetricsAccumulator()

    torch.set_grad_enabled(False)
    runner.model.eval()

    for batch in tqdm(valid_loader, desc='Running Eval', total=num_batches,
                      disable=not is_master, leave=False):
        loss, metrics = runner.forward(batch)  # type: ignore
        accumulator.update(loss.item(), {name: value.item() for name, value in metrics.items()})

    # Reduce loss across all processes if multiprocessing
    eval_loss = utils.reduce_scalar(accumulator.final_loss())
    metrics = {name: utils.reduce_scalar(value)
               for name, value in accumulator.final_metrics().items()}

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"
    for name, value in metrics.items():
        print_str += f"[{name.capitalize()}: {value:.5g}]"

    metrics['loss'] = eval_loss
    if viz is not None:
        viz.log_metrics(metrics, "val", getattr(runner, 'global_step', epoch_id))

    logger.info(print_str)

    return eval_loss, metrics


def run_eval_epoch(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   is_master: bool = True,
                   save_callback: typing.Optional[typing.Sequence[typing.Callable]] = None) \
        -> typing.Dict[str, typing.List[typing.Any]]:
    num_batches = len(eval_loader)
    accumulator = utils.MetricsAccumulator()
    torch.set_grad_enabled(False)
    runner.model.eval()

    save_outputs = []

    for batch in tqdm(eval_loader, desc='Evaluating split val', total=num_batches,
                      disable=not is_master):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True)  # type: ignore
        accumulator.update(
            loss.item(), {name: value.item() for name, value in metrics.items()})
        if save_callback is not None:
            to_save = dict(ChainMap(
                *(callback(runner.model, batch, outputs) for callback in save_callback)))
            save_outputs.append(to_save)

    eval_loss = utils.reduce_scalar(accumulator.final_loss())
    final_print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"
    for name, value in accumulator.final_metrics().items():
        value = utils.reduce_scalar(value)
        final_print_str += f"[{name.capitalize()}: {value:.5g}]"
    logger.info(final_print_str)

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
              tokenizer: str = 'amino_acid',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False) -> None:
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    exp_name = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir) / exp_name

    if is_master:
        # save all the hidden parameters.
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'config.json').open('w') as f:
            json.dump(input_args, f)

    utils.barrier_if_distributed()
    utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)

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

    model = models.get(model_type, task, model_config_file, from_pretrained)
    optimizer = utils.setup_optimizer(model, learning_rate)
    viz = visualization.get(log_dir, exp_name, local_rank, debug=debug)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    if fp16:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2", loss_scale="dynamic", master_weights=True)
        _amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    if resume_from_checkpoint:
        assert from_pretrained is not None
        checkpoint = torch.load(
            os.path.join(from_pretrained, 'checkpoint.bin'), map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if fp16:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved in zip(amp.master_params(optimizer), checkpoint['master params']):
                param.data.copy_(saved.data)
            amp.load_state_dict(checkpoint['amp'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        num_train_epochs -= start_epoch
    else:
        start_epoch = 0

    if local_rank != -1:
        # model = DDP(model)
        flat_dist_call([param.data for param in model.parameters()],
                       torch.distributed.broadcast, (0,))
    elif n_gpu > 1:
        model = nn.DataParallel(model)  # type: ignore

    runner = BackwardRunner(
        model, optimizer, scheduler, gradient_accumulation_steps,
        device, n_gpu, fp16, max_grad_norm, local_rank)

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
    utils.barrier_if_distributed()
    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        for epoch_id in range(start_epoch, num_train_epochs):
            run_train_epoch(epoch_id, train_loader, runner,
                            viz, num_log_iter, gradient_accumulation_steps)
            if not no_eval:
                val_loss, _ = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    num_epochs_no_improvement = 0
                else:
                    num_epochs_no_improvement += 1

            # Save trained model
            if do_save(epoch_id, num_epochs_no_improvement):
                logger.info("** ** * Saving trained model ** ** * ")
                # Only save the model itself
                output_model_dir = save_path / f"pytorch_model_{epoch_id}"
                output_model_dir.mkdir()
                utils.save_state(output_model_dir, model, optimizer, scheduler, epoch_id)
                logger.info(f"Saving model checkpoint to {output_model_dir}")

            utils.barrier_if_distributed()
            if patience > 0 and num_epochs_no_improvement >= patience:
                logger.info(f"Finished training at epoch {epoch_id} because no "
                            f"improvement for {num_epochs_no_improvement} epochs.")
                logger.log(35, f"Best Val Loss: {best_val_loss}")
                if local_rank != -1:
                    # If you're distributed, raise this error. It sends a signal to
                    # the master process which lets it kill other processes and terminate
                    # without actually reporting an error. See utils/distributed_utils.py
                    # for the signal handling code.
                    raise errors.EarlyStopping
                else:
                    break
    logger.info(f"Finished training after {num_train_epochs} epochs.")
    if not no_eval:
        logger.log(35, f"Best Val Loss: {best_val_loss}")
