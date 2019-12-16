from typing import Dict, Type, Callable, Optional, Union
from torch.utils.data import Dataset
from .models.modeling_utils import ProteinModel
from pathlib import Path

PathType = Union[str, Path]


class TAPETaskSpec:
    """
    Attributes
    ----------
    name (str):
        The name of the TAPE task
    dataset (Type[Dataset]):
        The dataset used in the TAPE task
    num_labels (int):
        number of labels used if this is a classification task
    models (Dict[str, ProteinModel]):
        The set of models that can be used for this task. Default: {}.
    """

    def __init__(self,
                 name: str,
                 dataset: Type[Dataset],
                 num_labels: int = -1,
                 models: Optional[Dict[str, Type[ProteinModel]]] = None):
        self.name = name
        self.dataset = dataset
        self.num_labels = num_labels
        self.models = models if models is not None else {}

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
    metric_name_mapping: Dict[str, Callable] = {}

    @classmethod
    def register_task(cls,
                      task_name: str,
                      num_labels: int = -1,
                      dataset: Optional[Type[Dataset]] = None,
                      models: Optional[Dict[str, Type[ProteinModel]]] = None):
        """ Register a a new TAPE task. This creates a new TAPETaskSpec.

        Args:

            task_name (str): The name of the TAPE task.
            num_labels (int): Number of labels used if this is a classification task. If this
                is not a classification task, simply leave the default as -1.
            dataset (Type[Dataset]): The dataset used in the TAPE task.
            models (Optional[Dict[str, ProteinModel]]): The set of models that can be used for
                this task. If you do not pass this argument, you can register models to the task
                later by using `registry.register_task_model`. Default: {}.

        Examples:

        There are two ways of registering a new task. First, one can define the task by simply
        declaring all the components, and then calling the register method, like so:

            class SecondaryStructureDataset(Dataset):
                ...

            class ProteinBertForSequenceToSequenceClassification():
                ...

            registry.register_task(
                'secondary_structure', 3, SecondaryStructureDataset,
                {'transformer': ProteinBertForSequenceToSequenceClassification})

        This will register a new task, 'secondary_structure', with a single model. More models
        can be added with `registry.register_task_model`. Alternatively, this can be used as a
        decorator:

            @registry.regsiter_task('secondary_structure', 3)
            class SecondaryStructureDataset(Dataset):
                ...

            @registry.register_task_model('secondary_structure', 'transformer')
            class ProteinBertForSequenceToSequenceClassification():
                ...

        These two pieces of code are exactly equivalent, in terms of the resulting registry
        state.

        """
        if dataset is not None:
            if models is None:
                models = {}
            task_spec = TAPETaskSpec(task_name, dataset, num_labels, models)
            return cls.register_task_spec(task_name, task_spec).dataset
        else:
            return lambda dataset: cls.register_task(task_name, num_labels, dataset, models)

    @classmethod
    def register_task_spec(cls, task_name: str, task_spec: Optional[TAPETaskSpec] = None):
        """ Registers a task_spec directly. If you find it easier to actually create a
            TAPETaskSpec manually, and then register it, feel free to use this method,
            but otherwise it is likely easier to use `registry.register_task`.
        """
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
        r"""Register a specific model to a task with the provided model name.
            The task must already be in the registry - you cannot register a
            model to an unregistered task.

        Args:
            task_name (str): Name of task to which to register the model.
            model_name (str): Name of model to use when registering task, this
                is the name that you will use to refer to the model on the
                command line.
            model_cls (Type[ProteinModel]): The model to register.

        Examples:

        As with `registry.register_task`, this can both be used as a regular
        python function, and as a decorator. For example this:

            class ProteinBertForSequenceToSequenceClassification():
                ...
            registry.register_task_model(
                'secondary_structure', 'transformer',
                ProteinBertForSequenceToSequenceClassification)

        and as a decorator:

            @registry.register_task_model('secondary_structure', 'transformer')
            class ProteinBertForSequenceToSequenceClassification():
                ...

        are both equivalent.
        """
        if task_name not in cls.task_name_mapping:
            raise KeyError(
                f"Tried to register a task model for an unregistered task: {task_name}. "
                f"Make sure to register the task {task_name} first.")
        return cls.task_name_mapping[task_name].register_model(model_name, model_cls)

    @classmethod
    def register_metric(cls, name: str) -> Callable[[Callable], Callable]:
        r"""Register a metric to registry with key 'name'

        Args:
            name: Key with which the metric will be registered.

        Usage::
            from tape.registry import registry

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
    def get_task_spec(cls, name: str) -> TAPETaskSpec:
        return cls.task_name_mapping[name]

    @classmethod
    def get_metric(cls, name: str) -> Callable:
        return cls.metric_name_mapping[name]

    @classmethod
    def get_task_model(cls,
                       model_name: str,
                       task_name: str,
                       config_file: Optional[PathType] = None,
                       load_dir: Optional[PathType] = None) -> ProteinModel:
        """ Create a TAPE task model, either from scratch or from a pretrained model.
            This is mostly a helper function that evaluates the if statements in a
            sensible order if you pass all three of the arguments.
        Args:
            model_name (str): Which type of model to create (e.g. transformer, unirep, ...)
            task_name (str): The TAPE task for which to create a model
            config_file (str, optional): A json config file that specifies hyperparameters
            load_dir (str, optional): A save directory for a pretrained model
        Returns:
            model (ProteinModel): A TAPE task model
        """
        task_spec = registry.get_task_spec(task_name)
        model_cls = task_spec.get_model(model_name)

        if load_dir is not None:
            model = model_cls.from_pretrained(load_dir, num_labels=task_spec.num_labels)
        else:
            config_class = model_cls.config_class
            if config_file is not None:
                config = config_class.from_json_file(config_file)
            else:
                config = config_class()
            config.num_labels = task_spec.num_labels
            model = model_cls(config)
        return model


registry = Registry()
