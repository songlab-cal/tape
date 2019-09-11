import torch.nn as nn
from pytorch_transformers.modeling_utils import PreTrainedModel
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers.modeling_bert import BertModel

from tape_pytorch.registry import registry


TransformerConfig = BertConfig
Transformer = BertModel
registry.register_model('transformer')(Transformer)


@registry.register_model('lstm')
class LSTM(PreTrainedModel):
    pass


@registry.register_model('unirep')
class UniRep(PreTrainedModel):
    pass


@registry.register_model('bepler')
class Bepler(PreTrainedModel):
    pass
