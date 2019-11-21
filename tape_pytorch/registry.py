from typing import Dict, Type, Callable, Optional, NamedTuple, List, Any
from torch.utils.data import Dataset
from protein_models import ProteinModel


class TAPETaskSpec(NamedTuple):
    """
    Attributes
    ----------
    name (str):
        The name of the TAPE task
    dataset (Type[Dataset]):
        The dataset used in the TAPE task
    collate_fn (Callable[[List], Dict[str, Any]]):
        The collate function used to return batches for this task
    num_labels (int):
        number of labels used if this is a classification task
    models (Dict[str, ProteinModel]):
        The set of models that can be used for this task. Default: {}.
    """

    name: str
    dataset: Type[Dataset]
    collate_fn: Type[Callable[[List], Dict[str, Any]]]
    num_labels: int = -1
    models: Dict[str, ProteinModel] = {}

    def register_model(self, model_name: str, model_cls: Optional[Type[ProteinModel]] = None):
        if model_cls is not None:
            if model_name in self.models:
                raise KeyError(
                    f"A model with name '{model_name}' is already registered for this task")
            self.models[model_name] = model_cls
            return model_cls
        else:
            return lambda model_cls: self.register_model(model_name, model_cls)

    def get_model(self, model_name: str) -> Type[ProteinModel]:
        return self.models[model_name]


class Registry:
    r"""Class for registry object which acts as the
    central repository for TAPE."""

    task_name_mapping: Dict[str, TAPETaskSpec] = {}
    tokenizer_name_mapping: Dict[str, Type] = {}
    callback_name_mapping: Dict[str, Callable] = {}
    metric_name_mapping: Dict[str, Callable] = {}

    @classmethod
    def register_task(cls,
                      task_name: str,
                      dataset: Type[Dataset],
                      collate_fn: Type[Callable[[List], Dict[str, Any]]],
                      num_labels: int = -1,
                      models: Optional[Dict[str, ProteinModel]] = None) -> TAPETaskSpec:
        if models is None:
            models = {}
        task_spec = TAPETaskSpec(task_name, dataset, collate_fn, num_labels, models)
        return cls.register_task_spec(task_name, task_spec)

    @classmethod
    def register_task_spec(cls, task_name: str, task_spec: Optional[TAPETaskSpec] = None):
        if task_spec is not None:
            if task_name in cls.task_name_mapping:
                raise KeyError(f"A task with name '{task_name}' is already registered")
            cls.task_name_mapping[task_name] = task_spec
            return task_spec
        else:
            return lambda task_spec: cls.register_task_spec(task_name, task_spec)

    @classmethod
    def register_task_model(cls,
                            task_name: str,
                            model_name: str,
                            model_cls: Optional[Type[ProteinModel]] = None):
        r"""Register a task model to registry with key 'name'

        Args:
            name: Key with which the task model will be registered.

        Usage::
            from tape_pytorch.registry import registry
            import torch.nn as nn

            @registry.register_task_model('fluorescence')
            @registry.register_task_model('stability')
            class SequenceToFloatModel(nn.Module):
                ...
        """
        if task_name not in cls.task_name_mapping:
            raise KeyError(
                f"Tried to register a task model for an unregistered task. "
                f"Make sure to register the task {task_name} first.")
        return cls.task_name_mapping[task_name].register_model(model_name, model_cls)

    @classmethod
    def register_callback(cls, name: str) -> Callable[[Callable], Callable]:
        r"""Register a callback to registry with key 'name'

        Args:
            name: Key with which the callback will be registered.

        Usage::
            from tape_pytorch.registry import registry

            @registry.register_callback('save_fluorescence')
            def save_float_prediction(inputs, outputs):
                ...
        """

        def wrap(fn: Callable) -> Callable:
            assert callable(fn), "All callbacks must be callable"
            cls.callback_name_mapping[name] = fn
            return fn

        return wrap

    @classmethod
    def register_metric(cls, name: str) -> Callable[[Callable], Callable]:
        r"""Register a metric to registry with key 'name'

        Args:
            name: Key with which the metric will be registered.

        Usage::
            from tape_pytorch.registry import registry

            @registry.register_metric('mse')
            def mean_squred_error(inputs, outputs):
                ...
        """

        def wrap(fn: Callable) -> Callable:
            assert callable(fn), "All metrics must be callable"
            cls.metric_name_mapping[name] = fn
            return fn

        return wrap

    @classmethod
    def register_tokenizer(cls, name: str) -> Callable[[Type], Type]:
        r"""Register a tokenizer to registry with key 'name'

        Args:
            name: Key with which the tokenizer will be registered.

        Usage::
            from tape_pytorch.registry import registry
            from tape_pytorch.tokenizers import TAPETokenizer

            @registry.register('bpe')
            class BPETokenizer(TAPETokenizer):
                ...
        """

        def wrap(tokenizer_cls: Type[Callable]) -> Type[Callable]:
            from tape_pytorch.tokenizers import TAPETokenizer
            assert issubclass(tokenizer_cls, TAPETokenizer), \
                "All collate_fn must inherit tape_pytorch.tokenizers.TAPETokenizer"
            cls.tokenizer_name_mapping[name] = tokenizer_cls
            return tokenizer_cls

        return wrap

    @classmethod
    def get_task_spec(cls, name: str) -> TAPETaskSpec:
        return cls.task_name_mapping[name]

    @classmethod
    def get_callback(cls, name: str) -> Callable:
        return cls.callback_name_mapping[name]

    @classmethod
    def get_metric(cls, name: str) -> Callable:
        return cls.metric_name_mapping[name]

    @classmethod
    def get_tokenizer_class(cls, name: str) -> Type:
        return cls.tokenizer_name_mapping[name]


registry = Registry()
