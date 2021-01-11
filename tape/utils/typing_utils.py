from typing import Union, Dict
from pathlib import Path
import torch


PathLike = Union[str, Path]
TensorDict = Dict[str, torch.Tensor]
