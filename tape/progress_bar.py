from typing import Dict, Iterable, Tuple, Any
import logging
from timeit import default_timer as timer
from datetime import timedelta
from copy import copy
import contextlib

from pytorch_lightning.callbacks.progress import ProgressBarBase


class MetricsAccumulator:

    def __init__(self, smoothing: float = 0.95):
        self._smoothing = smoothing
        self._metricstmp: Dict[str, float] = {}
        self._last_updated: Dict[str, int] = {}
        self._completed_batches = 0

    @property
    def smoothing(self) -> float:
        return self.smoothing

    @property
    def metrics(self) -> Dict[str, float]:
        return copy(self._metricstmp)

    def _update_value(self, name: str, value: float) -> None:
        if name.startswith('max') or name.startswith('min'):
            self._metricstmp[name] = value
        elif name in self._metricstmp:
            time = self._completed_batches - self._last_updated[name]
            weight = self._smoothing ** time
            self._metricstmp[name] = \
                weight * self._metricstmp[name] + (1 - weight) * value
            self._last_updated[name] = self._completed_batches
        else:
            self._metricstmp[name] = value
            self._last_updated[name] = self._completed_batches

    def update(self, metrics: Dict[str, Any]) -> None:
        for name, value in metrics.items():
            with contextlib.suppress(ValueError):
                value = float(value)
                self._update_value(name, value)
        self._completed_batches += 1

    def items(self):
        return self._metricstmp.items()


class LoggingProgressBar(ProgressBarBase):

    def __init__(self, refresh_rate: int = 1, smoothing: float = 0.95):
        super().__init__()
        self._enabled = True
        self._refresh_rate = refresh_rate
        self._smoothing = smoothing
        self.train_logger = logging.getLogger('train')
        self.val_logger = logging.getLogger('val')
        self.test_logger = logging.getLogger('test')

        self._time = timer()
        self._metrics = MetricsAccumulator(smoothing)
        self._completed_batches = 0

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    @property
    def epoch(self) -> int:
        return self.trainer.current_epoch

    @property
    def global_step(self) -> int:
        return self.trainer.global_step

    @property
    def total_train_steps(self) -> int:
        if self.trainer.fast_dev_run:
            num_batches = 1
        else:
            num_batches = min(
                self.trainer.num_training_batches * self.trainer.max_epochs,
                self.trainer.max_steps)
        return num_batches

    @property
    def completed_batches(self) -> int:
        return self._completed_batches

    @property
    def metrics(self) -> MetricsAccumulator:
        return self._metrics

    def prefix(self) -> str:
        return f"[Ep: {self.epoch + 1}][Step: {self.global_step + 1}]"

    def suffix(self) -> str:
        elapsed_time = timer() - self._time
        time_per_batch = elapsed_time / self.completed_batches
        num_batches_remaining = self.total_train_steps - self.completed_batches
        est_time_remaining = int(num_batches_remaining * time_per_batch)
        elapsed = timedelta(seconds=int(elapsed_time))
        left = timedelta(seconds=est_time_remaining)
        batch_per_second = self.completed_batches / elapsed_time
        return f"[{elapsed}<{left}][batch/s: {batch_per_second:.2f}]"

    def infix(self, metrics: Iterable[Tuple[str, float]]) -> str:
        return ''.join(f"[{name}: {value:.3g}]"
                       for name, value in metrics
                       if not isinstance(value, str))

    def increment_completed_batches(self) -> None:
        self._completed_batches += 1

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def on_batch_end(self, trainer, pl_module):
        super().on_batch_end(trainer, pl_module)
        self.increment_completed_batches()
        if self.is_enabled:
            self.metrics.update(trainer.progress_bar_dict)
            if self.completed_batches % self.refresh_rate == 0:
                msg = self.prefix() + self.infix(self.metrics.items()) + self.suffix()
                self.train_logger.info(msg)

    def on_epoch_start(self, trainer, pl_module):
        super().on_epoch_start(trainer, pl_module)

    def on_init_end(self, trainer):
        super().on_init_end(trainer)

    def on_test_batch_end(self, trainer, pl_module):
        super().on_test_batch_end(trainer, pl_module)

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)
