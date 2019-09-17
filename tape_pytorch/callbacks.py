from typing import Sequence
import torch
from tape_pytorch.registry import registry


@registry.register_callback('save_fluorescence')
@registry.register_callback('save_stability')
def save_float_prediction(inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]):
    inputs = tuple(t.cpu().numpy() for t in inputs)
    outputs = tuple(t.cpu().numpy() for t in outputs)

    sequence, mask, target = inputs
    loss, prediction = outputs

    return target, prediction
