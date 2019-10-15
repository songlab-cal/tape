import typing
from pathlib import Path

from .base_models import Bepler  # noqa: F401
from .base_models import LSTM  # noqa: F401
from .base_models import Transformer  # noqa: F401
from .base_models import UniRep  # noqa: F401
from .resnet import ResNet  # noqa: F401
from .task_models import TAPEConfig  # noqa: F401
from .task_models import TAPEPreTrainedModel  # noqa: F401
from .task_models import MaskedLMModel  # noqa: F401
from .task_models import FloatPredictModel  # noqa: F401
from .task_models import SequenceClassificationModel  # noqa: F401
from .task_models import RemoteHomologyModel  # noqa: F401
from .task_models import SequenceToSequenceClassificationModel  # noqa: F401
from .task_models import SS3ClassModel  # noqa: F401

from tape_pytorch.registry import registry


def from_pretrained(task: str, load_dir: typing.Union[str, Path]) -> TAPEPreTrainedModel:
    if task not in registry.task_model_name_mapping:
        raise ValueError(f"Unrecognized task: {task}")
    model_cls = registry.get_task_model_class(task)
    model = model_cls.from_pretrained(Path(load_dir))

    return model


def from_config(task: str, model_config_file: typing.Union[str, Path]) -> TAPEPreTrainedModel:
    if task not in registry.task_model_name_mapping:
        raise ValueError(f"Unrecognized task: {task}")
    model_cls = registry.get_task_model_class(task)
    config = TAPEConfig.from_json_file(model_config_file)
    model = model_cls(config)

    return model


def from_model_type(task: str, base_model_type: str) -> TAPEPreTrainedModel:
    if task not in registry.task_model_name_mapping:
        raise ValueError(f"Unrecognized task: {task}")
    model_cls = registry.get_task_model_class(task)
    base_config = registry.get_model_class(base_model_type).config_class()
    config = TAPEConfig(base_config, base_model=base_model_type)
    model = model_cls(config)

    return model
