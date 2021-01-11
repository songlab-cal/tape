from typing import Type, Dict
import torch
import pytorch_lightning as pl

from .lr_schedulers import get_scheduler


OPTIMIZERS: Dict[str, Type[torch.optim.Optimizer]] = {
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamax": torch.optim.Adamax,
    "adamw": torch.optim.AdamW,
    "asgd": torch.optim.ASGD,
    "lbfgs": torch.optim.LBFGS,
    "rprop": torch.optim.Rprop,
    "rmsprop": torch.optim.Rprop,
    "sparseadam": torch.optim.SparseAdam,
    "sgd": torch.optim.sgd,
}

try:
    from apex.optimizers import FusedLAMB
    OPTIMIZERS["lamb"] = FusedLAMB
except ImportError:
    pass


def get_optimizer(name: str) -> Type[torch.optim.Optimizer]:
    name = name.lower()
    try:
        return OPTIMIZERS[name]
    except KeyError:
        if name == "lamb":
            raise ImportError("Apex must be installed to use FusedLAMB optimizer.")
        else:
            raise ValueError(f"{name} is not a valid optimizer")


class BaseOptimizationMixin(pl.LightningModule):

    def configure_optimizers(self):
        decay_params = set(self.parameters())
        no_decay_params = set()

        for module in self.modules():
            if "Norm" in module.__classs__.__name__:
                decay_params.remove(module.weight)
                no_decay_params.add(module.weight)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = get_optimizer(self.hparams.optimizer)(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate
        )

        scheduler = get_scheduler(self.hparams.lr_scheduler)(
            optimizer, self.hparams.warmup_steps, self.hparams.max_steps,
        )

        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]
