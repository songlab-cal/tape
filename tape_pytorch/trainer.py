import logging

import torch
import torch.nn as nn

try:
    from apex import amp
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

logger = logging.getLogger(__name__)


class RunConfig:

    def __init__(self,
                 local_rank: int,
                 task: str,
                 data_dir: str = 'data',
                 num_train_epochs: int = 20,
                 num_log_iter: int = 20,
                 tokenizer: str = 'bpe',
                 cuda: bool = True,
                 use_fp16: bool = False,
                 max_grad_norm: float = 1.0,
                 gradient_accumulation_steps: int = 1,
                 train_batch_size: int = 1024,
                 seed: int = 42,
                 num_workers: int = 16,
                 warmup_steps: int = 10000,
                 debug: bool = False):

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

        if gradient_accumulation_steps < 1:
            raise ValueError(
                f"Invalid gradient_accumulation_steps parameter: "
                f"{gradient_accumulation_steps}, should be >= 1")

        if (use_fp16 or local_rank != -1) and not APEX_FOUND:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex "
                "to use distributed and fp16 training.")

        self.local_rank = local_rank
        self.task = task
        self.data_dir = data_dir
        self.num_train_epochs = num_train_epochs
        self.num_log_iter = num_log_iter
        self.tokenizer = tokenizer
        self.is_master = local_rank in (-1, 0)
        self.device = device
        self.n_gpu = n_gpu
        self.use_fp16 = use_fp16
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_batch_size = train_batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.warmup_steps = warmup_steps
        self.debug = debug

    def to_dict(self) -> dict:
        return {key: str(value) for key, value in self.__dict__.items()}

    @property
    def batch_size_per_gpu_forward(self) -> int:
        batch_size = float(self.train_batch_size)
        batch_size /= self.gradient_accumulation_steps
        batch_size /= self.n_gpu
        if self.local_rank != -1:
            batch_size /= torch.distributed.get_world_size()
        return int(batch_size)


class Trainer:

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 run_config: RunConfig):

        if run_config.use_fp16:
            if not APEX_FOUND:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex "
                    "to use distributed and fp16 training.")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_config = run_config
        self._global_step = 0

    def forward(self, batch) -> torch.Tensor:
        cuda_batch = tuple(
            t.cuda(device=self.run_config.device, non_blocking=True) for t in batch)
        outputs = self.model(*cuda_batch)
        loss = outputs[0]

        if self.run_config.n_gpu > 1:
            loss = loss.mean()

        return loss

    def backward(self, loss) -> None:
        if self.run_config.use_fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.run_config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()  # type: ignore
        self.optimizer.zero_grad()
        self._global_step += 1

    @property
    def global_step(self) -> int:
        return self._global_step
