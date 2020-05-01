from . import datasets  # noqa: F401
from . import metrics  # noqa: F401
from .tokenizers import TAPETokenizer  # noqa: F401
from .models.modeling_utils import ProteinModel
from .models.modeling_utils import ProteinConfig

import sys
from pathlib import Path
import importlib
import pkgutil

__version__ = '0.4'


# Import all the models and configs
for _, name, _ in pkgutil.iter_modules([str(Path(__file__).parent / 'models')]):
    imported_module = importlib.import_module('.models.' + name, package=__name__)
    for name, cls in imported_module.__dict__.items():
        if isinstance(cls, type) and \
                (issubclass(cls, ProteinModel) or issubclass(cls, ProteinConfig)):
            setattr(sys.modules[__name__], name, cls)
