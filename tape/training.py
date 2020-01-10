import typing
import os
import logging
from timeit import default_timer as timer
import json
from pathlib import Path
import inspect
import pickle as pkl

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .optimization import WarmupLinearSchedule

from . import utils
from . import errors
from . import visualization
from .registry import registry
from .models.modeling_utils import ProteinModel

try:
    from apex import amp
    import amp_C
    import apex_C
    from apex.amp import _amp_state
    from apex.parallel.distributed import flat_dist_call
    from apex.parallel.distributed import DistributedDataParallel as DDP
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

logger = logging.getLogger(__name__)

MetricsDict = typing.Dict[str, float]
LossAndMetrics = typing.Tuple[float, MetricsDict]
OutputDict = typing.Dict[str, typing.Any]


class ForwardRunner:

    def __init__(self,
                 model: ProteinModel,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1):

        self.model = model
        self.device = device
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.local_rank = local_rank

        forward_arg_keys = inspect.getfullargspec(model.forward).args
        forward_arg_keys = forward_arg_keys[1:]  # remove self argument
        self._forward_arg_keys = forward_arg_keys
        assert 'input_ids' in self._forward_arg_keys

    def initialize_distributed_model(self):
        if self.local_rank != -1:
            if not self.fp16:
                self.model = DDP(self.model)
            else:
                flat_dist_call([param.data for param in self.model.parameters()],
                               torch.distributed.broadcast, (0,))
        elif self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self,
                batch: typing.Dict[str, torch.Tensor],
                return_outputs: bool = False,
                no_loss: bool = False):
        # Filter out batch items that aren't used in this model
        # Requires that dataset keys match the forward args of the model
        # Useful if some elements of the data are only used by certain models
        # e.g. PSSMs / MSAs and other evolutionary data
        batch = {name: tensor for name, tensor in batch.items()
                 if name in self._forward_arg_keys}
        if self.device.type == 'cuda':
            batch = {name: tensor.cuda(device=self.device, non_blocking=True)
                     for name, tensor in batch.items()}

        outputs = self.model(**batch)

        if no_loss:
            return outputs

        if isinstance(outputs[0], tuple):
            # model also returned metrics
            loss, metrics = outputs[0]
        else:
            # no metrics
            loss = outputs[0]
            metrics = {}

        if self.n_gpu > 1:  # pytorch DataDistributed doesn't mean scalars
            loss = loss.mean()
            metrics = {name: metric.mean() for name, metric in metrics.items()}

        if return_outputs:
            return loss, metrics, outputs
        else:
            return loss, metrics

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self


class BackwardRunner(ForwardRunner):

    def __init__(self,
                 model: ProteinModel,
                 optimizer: optim.Optimizer,  # type: ignore
                 gradient_accumulation_steps: int = 1,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1,
                 max_grad_norm: float = 1.0,
                 warmup_steps: int = 0,
                 num_train_optimization_steps: int = 1000000):

        super().__init__(model, device, n_gpu, fp16, local_rank)
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self._global_step = 0
        self._local_rank = local_rank
        self._overflow_buf = torch.cuda.IntTensor([0])  # type: ignore
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._delay_accumulation = fp16 and local_rank != -1

        self.scheduler = WarmupLinearSchedule(
            self.optimizer, warmup_steps, num_train_optimization_steps)

    def initialize_fp16(self):
        if self.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level="O2", loss_scale="dynamic",
                master_weights=True)
            _amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    def resume_from_checkpoint(self, checkpoint_dir: str) -> int:
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, 'checkpoint.bin'), map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.fp16:
            self.optimizer._lazy_init_maybe_master_weights()
            self.optimizer._amp_stash.lazy_init_called = True
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved in zip(
                    amp.master_params(self.optimizer), checkpoint['master params']):
                param.data.copy_(saved.data)
            amp.load_state_dict(checkpoint['amp'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch

    def save_state(self, save_directory: typing.Union[str, Path], epoch_id: int):
        save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir()
        else:
            assert save_directory.is_dir(), "Save path should be a directory"
        model_to_save = getattr(self.model, 'module', self.model)
        model_to_save.save_pretrained(save_directory)
        optimizer_state: typing.Dict[str, typing.Any] = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch_id}
        if APEX_FOUND:
            optimizer_state['master params'] = list(amp.master_params(self.optimizer))
            try:
                optimizer_state['amp'] = amp.state_dict()
            except AttributeError:
                pass
        torch.save(optimizer_state, save_directory / 'checkpoint.bin')

    def backward(self, loss) -> None:
        if not self._delay_accumulation:
            loss = loss / self.gradient_accumulation_steps
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer,
                                delay_overflow_check=self._delay_accumulation) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if self._local_rank == -1:
            self._step()
        elif not self.fp16:
            # TODO: Can you do this allreduce after accumulation also?
            self._step()
        else:
            self._step_distributed_fp16()

    def _step(self) -> None:
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()  # type: ignore
        self._global_step += 1

    def _step_distributed_fp16(self) -> None:
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
    runner.train()

    def make_log_str(step: int, time: float) -> str:
        ep_percent = epoch_id + step / len(train_loader)
        if runner.scheduler is not None:
            curr_lr = runner.scheduler.get_lr()[0]  # type: ignore
        else:
            curr_lr = runner.optimizer.param_groups[0]['lr']

        print_str = []
        print_str.append(f"[Ep: {ep_percent:.2f}]")
        print_str.append(f"[Iter: {runner.global_step}]")
        print_str.append(f"[Time: {time:5.2f}s]")
        print_str.append(f"[Loss: {accumulator.loss():.5g}]")

        for name, value in accumulator.metrics().items():
            print_str.append(f"[{name.capitalize()}: {value:.5g}]")

        print_str.append(f"[LR: {curr_lr:.5g}]")
        return ''.join(print_str)

    start_t = timer()
    for step, batch in enumerate(train_loader):
        loss, metrics = runner.forward(batch)  # type: ignore
        runner.backward(loss)
        accumulator.update(loss, metrics, step=False)
        if (step + 1) % gradient_accumulation_steps == 0:
            runner.step()
            viz.log_metrics(accumulator.step(), "train", runner.global_step)
            if runner.global_step % num_log_iter == 0:
                end_t = timer()
                logger.info(make_log_str(step, end_t - start_t))
                start_t = end_t

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
    runner.eval()

    for batch in tqdm(valid_loader, desc='Running Eval', total=num_batches,
                      disable=not is_master, leave=False):
        loss, metrics = runner.forward(batch)  # type: ignore
        accumulator.update(loss, metrics)

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


def _get_outputs_to_save(batch, outputs):
    targets = batch['targets'].cpu().numpy()
    outputs = outputs.cpu().numpy()
    protein_length = batch['protein_length'].sum(1).cpu().numpy()

    reshaped_output = []
    for target, output, plength in zip(targets, outputs, protein_length):
        output_slices = tuple(slice(1, plength - 1) if dim == protein_length.max() else
                              slice(0, dim) for dim in output.shape)
        output = output[output_slices]
        target = target[output_slices]

        reshaped_output.append((target, output))
    reshaped_output


def run_eval_epoch(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   is_master: bool = True) -> typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()

    save_outputs = []

    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True)  # type: ignore
        predictions = outputs[1].cpu().numpy()
        targets = batch['targets'].cpu().numpy()
        for pred, target in zip(predictions, targets):
            save_outputs.append({'prediction': pred, 'target': target})

    return save_outputs


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
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              model_config_file: typing.Optional[str] = None,
              data_dir: str = './data',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False) -> None:

    # SETUP AND LOGGING CODE #
    input_args = locals()
    device, n_gpu, is_master = utils.setup_distributed(
        local_rank, no_cuda)

    exp_dir = utils.get_expname(exp_name, task, model_type)
    save_path = Path(output_dir) / exp_dir

    if is_master:
        # save all the hidden parameters.
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'args.json').open('w') as f:
            json.dump(input_args, f)

    utils.barrier_if_distributed()
    utils.setup_logging(local_rank, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)

    train_dataset = utils.setup_dataset(task, data_dir, 'train', tokenizer)
    valid_dataset = utils.setup_dataset(task, data_dir, 'valid', tokenizer)
    train_loader = utils.setup_loader(
        train_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    model = model.to(device)
    optimizer = utils.setup_optimizer(model, learning_rate)
    viz = visualization.get(log_dir, exp_dir, local_rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}, "
        f"distributed_training: {local_rank != -1}, "
        f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        assert from_pretrained is not None
        start_epoch = runner.resume_from_checkpoint(from_pretrained)
    else:
        start_epoch = 0
    runner.initialize_distributed_model()

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_epochs)
    is_master = local_rank in (-1, 0)

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num epochs = %d", num_train_epochs)
    logger.info("  Num train steps = %d", num_train_optimization_steps)
    logger.info("  Num parameters = %d", num_trainable_parameters)

    best_val_loss = float('inf')
    num_evals_no_improvement = 0

    def do_save(epoch_id: int, num_evals_no_improvement: int) -> bool:
        if not is_master:
            return False
        if isinstance(save_freq, int):
            return ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs)
        else:
            return num_evals_no_improvement == 0

    utils.barrier_if_distributed()

    # ACTUAL TRAIN/EVAL LOOP #
    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        for epoch_id in range(start_epoch, num_train_epochs):
            run_train_epoch(epoch_id, train_loader, runner,
                            viz, num_log_iter, gradient_accumulation_steps)
            if eval_freq > 0 and (epoch_id + 1) % eval_freq == 0:
                val_loss, _ = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    num_evals_no_improvement = 0
                else:
                    num_evals_no_improvement += 1

            # Save trained model
            if do_save(epoch_id, num_evals_no_improvement):
                logger.info("** ** * Saving trained model ** ** * ")
                # Only save the model itself
                runner.save_state(save_path, epoch_id)
                logger.info(f"Saving model checkpoint to {save_path}")

            utils.barrier_if_distributed()
            if patience > 0 and num_evals_no_improvement >= patience:
                logger.info(f"Finished training at epoch {epoch_id} because no "
                            f"improvement for {num_evals_no_improvement} epochs.")
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
    if best_val_loss != float('inf'):
        logger.log(35, f"Best Val Loss: {best_val_loss}")


def run_eval(model_type: str,
             task: str,
             from_pretrained: str,
             split: str = 'test',
             batch_size: int = 1024,
             model_config_file: typing.Optional[str] = None,
             data_dir: str = './data',
             no_cuda: bool = False,
             seed: int = 42,
             tokenizer: str = 'iupac',
             num_workers: int = 8,
             debug: bool = False,
             metrics: typing.Tuple[str, ...] = (),
             log_level: typing.Union[str, int] = logging.INFO) -> typing.Dict[str, float]:

    local_rank = -1  # TAPE does not support torch.distributed.launch for evaluation
    device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
    utils.setup_logging(local_rank, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    pretrained_dir = Path(from_pretrained)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}")

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained)
    model = model.to(device)

    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()
    valid_dataset = utils.setup_dataset(task, data_dir, split, tokenizer)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        1, num_workers)

    metric_functions = [registry.get_metric(name) for name in metrics]
    save_outputs = run_eval_epoch(valid_loader, runner, is_master)
    target = [el['target'] for el in save_outputs]
    prediction = [el['prediction'] for el in save_outputs]

    metrics_to_save = {name: metric(target, prediction)
                       for name, metric in zip(metrics, metric_functions)}
    logger.info(''.join(f'{name}: {val}' for name, val in metrics_to_save.items()))

    with (pretrained_dir / 'results.pkl').open('wb') as f:
        pkl.dump((metrics_to_save, save_outputs), f)

    return metrics_to_save


def run_embed(model_type: str,
              data_file: str,
              out_file: str,
              from_pretrained: str,
              batch_size: int = 1024,
              model_config_file: typing.Optional[str] = None,
              full_sequence_embed: bool = False,
              no_cuda: bool = False,
              seed: int = 42,
              tokenizer: str = 'iupac',
              num_workers: int = 8,
              log_level: typing.Union[str, int] = logging.INFO) -> None:

    local_rank = -1  # TAPE does not support torch.distributed.launch for embedding
    device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
    utils.setup_logging(local_rank, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    logger.info(
        f"device: {device} "
        f"n_gpu: {n_gpu}")

    task_spec = registry.get_task_spec('embed')
    model = registry.get_task_model(
        model_type, task_spec.name, model_config_file, from_pretrained)
    model = model.to(device)
    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()
    runner.eval()
    torch.set_grad_enabled(False)

    dataset = task_spec.dataset(data_file, tokenizer=tokenizer)  # type: ignore
    valid_loader = utils.setup_loader(dataset, batch_size, local_rank, n_gpu, 1, num_workers)

    with utils.IncrementalNPZ(out_file) as npzfile:
        with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu):
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                outputs = runner.forward(batch, no_loss=True)
                ids = batch['ids']
                sequence_embed = outputs[0]
                pooled_embed = outputs[1]
                sequence_lengths = batch['input_mask'].sum(1)
                sequence_embed = sequence_embed.cpu().numpy()
                pooled_embed = pooled_embed.cpu().numpy()
                sequence_lengths = sequence_lengths.cpu().numpy()

                for seqembed, poolembed, length, protein_id in zip(
                        sequence_embed, pooled_embed, sequence_lengths, ids):
                    seqembed = seqembed[:length]
                    arrays = {'pooled': poolembed}
                    if not full_sequence_embed:
                        # avgpool across the sequence
                        arrays['avg'] = seqembed.mean(0)
                    else:
                        arrays['seq'] = seqembed
                    to_save = {protein_id: arrays}
                    npzfile.savez(**to_save)
