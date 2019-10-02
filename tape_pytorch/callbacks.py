from typing import Dict
import torch
import numpy as np
from tape_pytorch.registry import registry


def _get_sequence_lengths(input_ids: torch.Tensor) -> np.ndarray:
    sequence_lengths = torch.sum(input_ids != 0, 1)
    return sequence_lengths.cpu().numpy()


@registry.register_callback('save_predictions')
def save_default(model,
                 inputs: Dict[str, torch.Tensor],
                 outputs: Dict[str, torch.Tensor]):

    model = getattr(model, 'module', model)  # get around DataParallel wrapper

    target = inputs[model.TARGET_KEY].cpu().numpy()
    prediction = outputs[model.PREDICTION_KEY].cpu().numpy()

    if target.shape[-1] == 1:
        target = np.squeeze(target, -1)
    if prediction.shape[-1] == 1:
        prediction = np.squeeze(prediction, -1)

    if model.PREDICTION_IS_SEQUENCE:
        sequence_lengths = _get_sequence_lengths(inputs['input_ids'])
        target = [t[:seqlen] for t, seqlen in zip(target, sequence_lengths)]
        prediction = [p[:seqlen] for p, seqlen in zip(prediction, sequence_lengths)]
    else:
        target = [t for t in target]
        prediction = [p for p in prediction]

    return {model.TARGET_KEY: target, model.PREDICTION_KEY: prediction}


@registry.register_callback('save_embedding')
def save_embedding(model,
                   inputs: Dict[str, torch.Tensor],
                   outputs: Dict[str, torch.Tensor]):

    model = getattr(model, 'module', model)  # get around DataParallel wrapper
    sequence_embedding = outputs[model.SEQUENCE_EMBEDDING_KEY].cpu().numpy()
    pooled_embedding = outputs[model.POOLED_EMBEDDING_KEY].cpu().numpy()
    sequence_lengths = _get_sequence_lengths(inputs['input_ids'])
    sequence_embedding = [embed[:seqlen]
                          for embed, seqlen in zip(sequence_embedding, sequence_lengths)]
    pooled_embedding = [embed for embed in pooled_embedding]

    return {model.SEQUENCE_EMBEDDING_KEY: sequence_embedding,
            model.POOLED_EMBEDDING_KEY: pooled_embedding}
