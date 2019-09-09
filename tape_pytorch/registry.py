from typing import Dict, Type, Callable

import torch.nn as nn
from torch.utils.data import Dataset


class Registry:
    r"""Class for registry object which acts as the
    central repository for TAPE."""

    dataset_name_mapping: Dict[str, Type[Dataset]] = {}
    model_name_mapping: Dict[str, Type] = {}
    task_model_name_mapping: Dict[str, Type] = {}
    collate_fn_name_mapping: Dict[str, Type[Callable]] = {}
    tokenizer_name_mapping: Dict[str, Type] = {}

    @classmethod
    def register_dataset(cls, name: str) -> Callable[[Type[Dataset]], Type[Dataset]]:
        r"""Register a dataset to registry with key 'name'

        Args:
            name: Key with which the dataset will be registered.

        Usage::
            from tape_pytorch.registry import registry
            from torch.utils.data import Dataset

            @registry.register('fluorescence')
            class FluorescenceDataset(Dataset):
                ...
        """

        def wrap(task_cls: Type[Dataset]) -> Type[Dataset]:
            assert issubclass(task_cls, Dataset), \
                "All datasets must inherit torch Dataset class"
            cls.dataset_name_mapping[name] = task_cls
            return task_cls

        return wrap

    @classmethod
    def register_model(cls, name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the model will be registered.

        Usage::
            from tape_pytorch.registry import registry
            import torch.nn as nn

            @registry.register('lstm')
            class LSTM(nn.Module):
                ...
        """

        def wrap(model_cls: Type[nn.Module]) -> Type[nn.Module]:
            assert issubclass(model_cls, nn.Module), \
                "All models must inherit torch Module class"
            cls.model_name_mapping[name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def register_task_model(cls, name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
        r"""Register a task model to registry with key 'name'

        Args:
            name: Key with which the task model will be registered.

        Usage::
            from tape_pytorch.registry import registry
            import torch.nn as nn

            @registry.register('fluorescence')
            @registry.register('stability')
            class SequenceToFloatModel(nn.Module):
                ...
        """

        def wrap(model_cls: Type[nn.Module]) -> Type[nn.Module]:
            import torch.nn as nn
            assert issubclass(model_cls, nn.Module), \
                "All models must inherit torch Module class"
            cls.task_model_name_mapping[name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def register_collate_fn(cls, name: str) -> Callable[[Type[Callable]], Type[Callable]]:
        r"""Register a collate_fn to registry with key 'name'

        Args:
            name: Key with which the collate_fn will be registered.

        Usage::
            from tape_pytorch.registry import registry
            from tape_pytorch.datasets import PaddedBatch

            @registry.register('pfam')
            class PfamBatch(PaddedBatch):
                ...
        """

        def wrap(fn_cls: Type[Callable]) -> Type[Callable]:
            from tape_pytorch.datasets import PaddedBatch
            assert issubclass(fn_cls, PaddedBatch), \
                "All collate_fn must inherit tape_pytorch.datasets.PaddedBatch"
            cls.collate_fn_name_mapping[name] = fn_cls
            return fn_cls

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
    def get_dataset_class(cls, name: str) -> Type[Dataset]:
        return cls.dataset_name_mapping[name]

    @classmethod
    def get_model_class(cls, name: str) -> Type:
        return cls.model_name_mapping[name]

    @classmethod
    def get_task_model_class(cls, name: str) -> Type:
        return cls.task_model_name_mapping[name]

    @classmethod
    def get_collate_fn_class(cls, name: str) -> Type[Callable]:
        return cls.collate_fn_name_mapping[name]

    @classmethod
    def get_tokenizer_class(cls, name: str) -> Type:
        return cls.tokenizer_name_mapping[name]


registry = Registry()
