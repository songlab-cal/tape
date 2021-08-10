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


URL_PREFIX = "https://s3.amazonaws.com/songlabdata/proteindata/pytorch-models/"
LSTM_PRETRAINED_CONFIG_ARCHIVE_MAP: typing.Dict[str, str] = {}
LSTM_PRETRAINED_MODEL_ARCHIVE_MAP: typing.Dict[str, str] = {}


class ProteinLSTMConfig(ProteinConfig):
    pretrained_config_archive_map = LSTM_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size: int = 30,
                 input_size: int = 128,
                 hidden_size: int = 1024,
                 num_hidden_layers: int = 3,
                 hidden_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range


class ProteinLSTMLayer(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        return self.lstm(inputs)


class ProteinLSTMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scalar_reweighting = nn.Linear(2 * config.num_hidden_layers, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.scalar_reweighting(hidden_states).squeeze(2)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProteinLSTMEncoder(nn.Module):

    def __init__(self, config: ProteinLSTMConfig):
        super().__init__()
        forward_lstm = [ProteinLSTMLayer(config.input_size, config.hidden_size)]
        reverse_lstm = [ProteinLSTMLayer(config.input_size, config.hidden_size)]
        for _ in range(config.num_hidden_layers - 1):
            forward_lstm.append(ProteinLSTMLayer(
                config.hidden_size, config.hidden_size, config.hidden_dropout_prob))
            reverse_lstm.append(ProteinLSTMLayer(
                config.hidden_size, config.hidden_size, config.hidden_dropout_prob))
        self.forward_lstm = nn.ModuleList(forward_lstm)
        self.reverse_lstm = nn.ModuleList(reverse_lstm)
        self.output_hidden_states = config.output_hidden_states

    def forward(self, inputs, input_mask=None):
        all_forward_pooled = ()
        all_reverse_pooled = ()
        all_hidden_states = (inputs,)
        forward_output = inputs
        for layer in self.forward_lstm:
            forward_output, forward_pooled = layer(forward_output)
            all_forward_pooled = all_forward_pooled + (forward_pooled[0],)
            all_hidden_states = all_hidden_states + (forward_output,)

        reversed_sequence = self.reverse_sequence(inputs, input_mask)
        reverse_output = reversed_sequence
        for layer in self.reverse_lstm:
            reverse_output, reverse_pooled = layer(reverse_output)
            all_reverse_pooled = all_reverse_pooled + (reverse_pooled[0],)
            all_hidden_states = all_hidden_states + (reverse_output,)
        reverse_output = self.reverse_sequence(reverse_output, input_mask)

        output = torch.cat((forward_output, reverse_output), dim=2)
        pooled = all_forward_pooled + all_reverse_pooled
        pooled = torch.stack(pooled, 3).squeeze(0)
        outputs = (output, pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        return outputs  # sequence_embedding, pooled_embedding, (hidden_states)

    def reverse_sequence(self, sequence, input_mask):
        if input_mask is None:
            idx = torch.arange(sequence.size(1) - 1, -1, -1)
            reversed_sequence = sequence.index_select(1, idx, device=sequence.device)
        else:
            sequence_lengths = input_mask.sum(1)
            reversed_sequence = []
            for seq, seqlen in zip(sequence, sequence_lengths):
                idx = torch.arange(seqlen - 1, -1, -1, device=seq.device)
                seq = seq.index_select(0, idx)
                seq = F.pad(seq, [0, 0, 0, sequence.size(1) - seqlen])
                reversed_sequence.append(seq)
            reversed_sequence = torch.stack(reversed_sequence, 0)
        return reversed_sequence


class ProteinLSTMAbstractModel(ProteinModel):

    config_class = ProteinLSTMConfig
    pretrained_model_archive_map = LSTM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "lstm"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@registry.register_task_model('embed', 'lstm')
class ProteinLSTMModel(ProteinLSTMAbstractModel):

    def __init__(self, config: ProteinLSTMConfig):
        super().__init__(config)
        self.embed_matrix = nn.Embedding(config.vocab_size, config.input_size)
        self.encoder = ProteinLSTMEncoder(config)
        self.pooler = ProteinLSTMPooler(config)
        self.output_hidden_states = config.output_hidden_states
        self.init_weights()

    def forward(self, input_ids, input_mask=None):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        # fp16 compatibility
        embedding_output = self.embed_matrix(input_ids)
        outputs = self.encoder(embedding_output, input_mask=input_mask)
        sequence_output = outputs[0]
        pooled_outputs = self.pooler(outputs[1])

        outputs = (sequence_output, pooled_outputs) + outputs[2:]
        return outputs  # sequence_output, pooled_output, (hidden_states)


@registry.register_task_model('language_modeling', 'lstm')
class ProteinLSTMForLM(ProteinLSTMAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.lstm = ProteinLSTMModel(config)
        self.feedforward = nn.Linear(config.hidden_size, config.vocab_size)

        self.init_weights()

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None):

        outputs = self.lstm(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]

        forward_prediction, reverse_prediction = sequence_output.chunk(2, -1)
        forward_prediction = F.pad(forward_prediction[:, :-1], [0, 0, 1, 0])
        reverse_prediction = F.pad(reverse_prediction[:, 1:], [0, 0, 0, 1])
        prediction_scores = \
            self.feedforward(forward_prediction) + self.feedforward(reverse_prediction)
        prediction_scores = prediction_scores.contiguous()

        # add hidden states and if they are here
        outputs = (prediction_scores,) + outputs[2:]

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), targets.view(-1))
            outputs = (lm_loss,) + outputs

        # (loss), prediction_scores, seq_relationship_score, (hidden_states)
        return outputs


@registry.register_task_model('fluorescence', 'lstm')
@registry.register_task_model('stability', 'lstm')
class ProteinLSTMForValuePrediction(ProteinLSTMAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.lstm = ProteinLSTMModel(config)
        self.predict = ValuePredictionHead(config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.lstm(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('remote_homology', 'lstm')
class ProteinLSTMForSequenceClassification(ProteinLSTMAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.lstm = ProteinLSTMModel(config)
        self.classify = SequenceClassificationHead(
            config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.lstm(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('secondary_structure', 'lstm')
class ProteinLSTMForSequenceToSequenceClassification(ProteinLSTMAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.lstm = ProteinLSTMModel(config)
        self.classify = SequenceToSequenceClassificationHead(
            config.hidden_size * 2, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.lstm(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        amino_acid_class_scores = self.classify(sequence_output.contiguous())

        # add hidden states and if they are here
        outputs = (amino_acid_class_scores,) + outputs[2:]

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            classification_loss = loss_fct(
                amino_acid_class_scores.view(-1, self.config.num_labels),
                targets.view(-1))
            outputs = (classification_loss,) + outputs

        # (loss), prediction_scores, seq_relationship_score, (hidden_states)
        return outputs


@registry.register_task_model('contact_prediction', 'lstm')
class ProteinLSTMForContactPrediction(ProteinLSTMAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.lstm = ProteinLSTMModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):

        outputs = self.lstm(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
