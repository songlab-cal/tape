import logging
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_utils import ProteinConfig
from .modeling_utils import ProteinModel
from .modeling_utils import ValuePredictionHead
from .modeling_utils import SequenceClassificationHead
from .modeling_utils import SequenceToSequenceClassificationHead
from .modeling_utils import PairwiseContactPredictionHead
from ..registry import registry

logger = logging.getLogger(__name__)


class ProteinOneHotConfig(ProteinConfig):
    pretrained_config_archive_map: typing.Dict[str, str] = {}

    def __init__(self,
                 vocab_size: int,
                 initializer_range: float = 0.02,
                 use_evolutionary: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.use_evolutionary = use_evolutionary
        self.initializer_range = initializer_range


class ProteinOneHotAbstractModel(ProteinModel):

    config_class = ProteinOneHotConfig
    pretrained_model_archive_map: typing.Dict[str, str] = {}
    base_model_prefix = "onehot"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class ProteinOneHotModel(ProteinOneHotAbstractModel):

    def __init__(self, config: ProteinOneHotConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size

        # Note: this exists *solely* for fp16 support
        # There doesn't seem to be an easier way to check whether to use fp16 or fp32 training
        buffer = torch.tensor([0.])
        self.register_buffer('_buffer', buffer)

    def forward(self, input_ids, input_mask=None):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        sequence_output = F.one_hot(input_ids, num_classes=self.vocab_size)
        # fp16 compatibility
        sequence_output = sequence_output.type_as(self._buffer)
        input_mask = input_mask.unsqueeze(2).type_as(sequence_output)
        # just a bag-of-words for amino acids
        pooled_outputs = (sequence_output * input_mask).sum(1) / input_mask.sum(1)

        outputs = (sequence_output, pooled_outputs)
        return outputs


@registry.register_task_model('fluorescence', 'onehot')
@registry.register_task_model('stability', 'onehot')
class ProteinOneHotForValuePrediction(ProteinOneHotAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.onehot = ProteinOneHotModel(config)
        self.predict = ValuePredictionHead(config.vocab_size)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.onehot(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('remote_homology', 'onehot')
class ProteinOneHotForSequenceClassification(ProteinOneHotAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.onehot = ProteinOneHotModel(config)
        self.classify = SequenceClassificationHead(config.vocab_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.onehot(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('secondary_structure', 'onehot')
class ProteinOneHotForSequenceToSequenceClassification(ProteinOneHotAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.onehot = ProteinOneHotModel(config)
        self.classify = SequenceToSequenceClassificationHead(
            config.vocab_size, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.onehot(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('contact_prediction', 'onehot')
class ProteinOneHotForContactPrediction(ProteinOneHotAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.onehot = ProteinOneHotModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):

        outputs = self.onehot(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
