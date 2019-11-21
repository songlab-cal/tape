from typing import Union
from typing import Optional
from pathlib import Path

import protein_models
from tape_pytorch.registry import registry


PathType = Union[str, Path]

registry.register_task_model('mlm', 'transformer',
                             protein_models.ProteinBertForMaskedLM)
registry.register_task_model('secondary_structure', 'transformer',
                             protein_models.ProteinBertForSequenceToSequenceClassification)
registry.register_task_model('remote_homology', 'transformer',
                             protein_models.ProteinBertForSequenceClassification)
registry.register_task_model('fluorescence', 'transformer',
                             protein_models.ProteinBertForValuePrediction)
registry.register_task_model('stability', 'transformer',
                             protein_models.ProteinBertForValuePrediction)
registry.register_task_model('mlm', 'resnet',
                             protein_models.ProteinResNetForMaskedLM)
registry.register_task_model('secondary_structure', 'resnet',
                             protein_models.ProteinResNetForSequenceToSequenceClassification)
registry.register_task_model('remote_homology', 'resnet',
                             protein_models.ProteinResNetForSequenceClassification)
registry.register_task_model('fluorescence', 'resnet',
                             protein_models.ProteinResNetForValuePrediction)
registry.register_task_model('stability', 'resnet',
                             protein_models.ProteinResNetForValuePrediction)


def get(base_model: str,
        task: str,
        config_file: Optional[PathType] = None,
        load_dir: Optional[PathType] = None) -> protein_models.ProteinModel:
    """ Create a TAPE task model, either from scratch or from a pretrained model.
        This is mostly a helper function that evaluates the if statements in a
        sensible order if you pass all three of the arguments.

    Args:
        base_model (str): Which type of model to create (e.g. transformer, unirep, ...)
        task (str): The TAPE task for which to create a model
        config_file (str, optional): A json config file that specifies hyperparameters
        load_dir (str, optional): A save directory for a pretrained model

    Returns:
        model (ProteinModel): A TAPE task model
    """
    task_spec = registry.get_task_spec(task)
    model_cls = task_spec.get_model(base_model)

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
    model = model.cuda()
    return model
