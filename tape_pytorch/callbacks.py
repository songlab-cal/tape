from typing import Dict
import torch
from tape_pytorch.registry import registry


@registry.register_callback('save_default')
def save_default(model,
                 inputs: Dict[str, torch.Tensor],
                 outputs: Dict[str, torch.Tensor]):

    model = getattr(model, 'module', model)  # get around DataParallel wrapper
    target = inputs[model.TARGET_KEY].cpu().numpy()
    prediction = outputs[model.PREDICTION_KEY].cpu().numpy()

    return {model.TARGET_KEY: target, model.PREDICTION_KEY: prediction}


@registry.register_callback('save_embedding')
def save_embedding(model,
                   inputs: Dict[str, torch.Tensor],
                   outputs: Dict[str, torch.Tensor]):

    model = getattr(model, 'module', model)  # get around DataParallel wrapper
    sequence_embedding = outputs[model.SEQUENCE_EMBEDDING_KEY].cpu().numpy()
    pooled_embedding = outputs[model.POOLED_EMBEDDING_KEY].cpu().numpy()

    return {model.SEQUENCE_EMBEDDING_KEY: sequence_embedding,
            model.POOLED_EMBEDDING_KEY: pooled_embedding}
