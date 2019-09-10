import torch.nn as nn
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers.modeling_bert import BertModel

from tape_pytorch.registry import registry


TAPEConfig = BertConfig
Transformer = BertModel
registry.register_model('transformer')(Transformer)


@registry.register_model('resnet')
class ResNet(nn.Module):
    pass


@registry.register_model('lstm')
class LSTM(nn.Module):
    pass


@registry.register_model('unirep')
class UniRep(nn.Module):
    pass


@registry.register_model('bepler')
class Bepler(nn.Module):
    pass
