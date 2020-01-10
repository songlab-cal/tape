import typing
import logging
import torch
import torch.nn as nn

from .modeling_utils import ProteinConfig
from .modeling_utils import ProteinModel
from .modeling_utils import get_activation_fn
from .modeling_utils import MLMHead
from .modeling_utils import LayerNorm
from .modeling_utils import ValuePredictionHead
from .modeling_utils import SequenceClassificationHead
from .modeling_utils import SequenceToSequenceClassificationHead
from .modeling_utils import PairwiseContactPredictionHead
from ..registry import registry

logger = logging.getLogger(__name__)

RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP: typing.Dict[str, str] = {}
RESNET_PRETRAINED_MODEL_ARCHIVE_MAP: typing.Dict[str, str] = {}


class ProteinResNetConfig(ProteinConfig):
    pretrained_config_archive_map = RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size: int = 30,
                 hidden_size: int = 512,
                 num_hidden_layers: int = 30,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class MaskedConv1d(nn.Conv1d):

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x)


class ProteinResNetLayerNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm = LayerNorm(config.hidden_size)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ProteinResNetBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.conv1 = MaskedConv1d(
            config.hidden_size, config.hidden_size, 3, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(config.hidden_size)
        self.bn1 = ProteinResNetLayerNorm(config)
        self.conv2 = MaskedConv1d(
            config.hidden_size, config.hidden_size, 3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm1d(config.hidden_size)
        self.bn2 = ProteinResNetLayerNorm(config)
        self.activation_fn = get_activation_fn(config.hidden_act)

    def forward(self, x, input_mask=None):
        identity = x

        out = self.conv1(x, input_mask)
        out = self.bn1(out)
        out = self.activation_fn(out)

        out = self.conv2(out, input_mask)
        out = self.bn2(out)

        out += identity
        out = self.activation_fn(out)

        return out


class ProteinResNetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, embed_dim, padding_idx=0)
        inverse_frequency = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.register_buffer('inverse_frequency', inverse_frequency)

        self.layer_norm = LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        words_embeddings = self.word_embeddings(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length - 1, -1, -1.0,
            dtype=words_embeddings.dtype,
            device=words_embeddings.device)
        sinusoidal_input = torch.ger(position_ids, self.inverse_frequency)
        position_embeddings = torch.cat([sinusoidal_input.sin(), sinusoidal_input.cos()], -1)
        position_embeddings = position_embeddings.unsqueeze(0)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ProteinResNetPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_weights = nn.Linear(config.hidden_size, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        attention_scores = self.attention_weights(hidden_states)
        if mask is not None:
            attention_scores += -10000. * (1 - mask)
        attention_weights = torch.softmax(attention_scores, -1)
        weighted_mean_embedding = torch.matmul(
            hidden_states.transpose(1, 2), attention_weights).squeeze(2)
        pooled_output = self.dense(weighted_mean_embedding)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ResNetEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [ProteinResNetBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, input_mask=None):
        all_hidden_states = ()
        for layer_module in self.layer:
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states, input_mask)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        return outputs


class ProteinResNetAbstractModel(ProteinModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = ProteinResNetConfig
    pretrained_model_archive_map = RESNET_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "resnet"

    def __init__(self, config):
        super().__init__(config)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
        # elif isinstance(module, ProteinResNetBlock):
            # nn.init.constant_(module.bn2.weight, 0)


@registry.register_task_model('embed', 'resnet')
class ProteinResNetModel(ProteinResNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = ProteinResNetEmbeddings(config)
        self.encoder = ResNetEncoder(config)
        self.pooler = ProteinResNetPooler(config)

        self.init_weights()

    def forward(self,
                input_ids,
                input_mask=None):
        if input_mask is not None and torch.any(input_mask != 1):
            extended_input_mask = input_mask.unsqueeze(2)
            # fp16 compatibility
            extended_input_mask = extended_input_mask.to(
                dtype=next(self.parameters()).dtype)
        else:
            extended_input_mask = None

        embedding_output = self.embeddings(input_ids)
        embedding_output = embedding_output.transpose(1, 2)
        if extended_input_mask is not None:
            extended_input_mask = extended_input_mask.transpose(1, 2)
        encoder_outputs = self.encoder(embedding_output, extended_input_mask)
        sequence_output = encoder_outputs[0]
        sequence_output = sequence_output.transpose(1, 2).contiguous()
        # sequence_output = encoder_outputs[0]
        if extended_input_mask is not None:
            extended_input_mask = extended_input_mask.transpose(1, 2)
        pooled_output = self.pooler(sequence_output, extended_input_mask)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs  # sequence_output, pooled_output, (hidden_states)


@registry.register_task_model('masked_language_modeling', 'resnet')
class ProteinResNetForMaskedLM(ProteinResNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.resnet = ProteinResNetModel(config)
        self.mlm = MLMHead(
            config.hidden_size, config.vocab_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.resnet.embeddings.word_embeddings)

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None):

        outputs = self.resnet(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.mlm(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('fluorescence', 'resnet')
@registry.register_task_model('stability', 'resnet')
class ProteinResNetForValuePrediction(ProteinResNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.resnet = ProteinResNetModel(config)
        self.predict = ValuePredictionHead(config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.resnet(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('remote_homology', 'resnet')
class ProteinResNetForSequenceClassification(ProteinResNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.resnet = ProteinResNetModel(config)
        self.classify = SequenceClassificationHead(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.resnet(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('secondary_structure', 'resnet')
class ProteinResNetForSequenceToSequenceClassification(ProteinResNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.resnet = ProteinResNetModel(config)
        self.classify = SequenceToSequenceClassificationHead(
            config.hidden_size, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.resnet(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('contact_prediction', 'resnet')
class ProteinResNetForContactPrediction(ProteinResNetAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.resnet = ProteinResNetModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):

        outputs = self.resnet(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
