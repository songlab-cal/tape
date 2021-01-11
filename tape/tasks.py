from typing import Union, Dict, List, Callable, Optional
import logging
from pathlib import Path
import tempfile
import tarfile
from argparse import ArgumentParser
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from tape.datasets import LMDBDataset
from tape.tokenizers import TAPETokenizer
from . import lr_schedulers
from .utils import http_get, pad_sequences

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parents[1] / "data"
