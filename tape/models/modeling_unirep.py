import logging
import typing
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .modeling_utils import ProteinConfig
from .modeling_utils import ProteinModel
from .modeling_utils import ValuePredictionHead
from .modeling_utils import SequenceClassificationHead
from .modeling_utils import SequenceToSequenceClassificationHead
from .modeling_utils import PairwiseContactPredictionHead
from ..registry import registry

logger = logging.getLogger(__name__)


URL_PREFIX = "https://s3.amazonaws.com/songlabdata/proteindata/pytorch-models/"
UNIREP_PRETRAINED_CONFIG_ARCHIVE_MAP: typing.Dict[str, str] = {
    'babbler-1900': URL_PREFIX + 'unirep-base-config.json'}
UNIREP_PRETRAINED_MODEL_ARCHIVE_MAP: typing.Dict[str, str] = {
    'babbler-1900': URL_PREFIX + 'unirep-base-pytorch_model.bin'}


class UniRepConfig(ProteinConfig):
    pretrained_config_archive_map = UNIREP_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size: int = 26,
                 input_size: int = 10,
                 hidden_size: int = 1900,
                 hidden_dropout_prob: float = 0.1,
                 layer_norm_eps: float = 1e-12,
                 initializer_range: float = 0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


class mLSTMCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        project_size = config.hidden_size * 4
        self.wmx = weight_norm(
            nn.Linear(config.input_size, config.hidden_size, bias=False))
        self.wmh = weight_norm(
            nn.Linear(config.hidden_size, config.hidden_size, bias=False))
        self.wx = weight_norm(
            nn.Linear(config.input_size, project_size, bias=False))
        self.wh = weight_norm(
            nn.Linear(config.hidden_size, project_size, bias=True))

    def forward(self, inputs, state):
        h_prev, c_prev = state
        m = self.wmx(inputs) * self.wmh(h_prev)
        z = self.wx(inputs) + self.wh(m)
        i, f, o, u = torch.chunk(z, 4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        c = f * c_prev + i * u
        h = o * torch.tanh(c)

        return h, c


class mLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlstm_cell = mLSTMCell(config)
        self.hidden_size = config.hidden_size

    def forward(self, inputs, state=None, mask=None):
        batch_size = inputs.size(0)
        seqlen = inputs.size(1)

        if mask is None:
            mask = torch.ones(batch_size, seqlen, 1, dtype=inputs.dtype, device=inputs.device)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(2)

        if state is None:
            zeros = torch.zeros(batch_size, self.hidden_size,
                                dtype=inputs.dtype, device=inputs.device)
            state = (zeros, zeros)

        steps = []
        for seq in range(seqlen):
            prev = state
            seq_input = inputs[:, seq, :]
            hx, cx = self.mlstm_cell(seq_input, state)
            seqmask = mask[:, seq]
            hx = seqmask * hx + (1 - seqmask) * prev[0]
            cx = seqmask * cx + (1 - seqmask) * prev[1]
            state = (hx, cx)
            steps.append(hx)

        return torch.stack(steps, 1), (hx, cx)


class UniRepAbstractModel(ProteinModel):

    config_class = UniRepConfig
    pretrained_model_archive_map = UNIREP_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "unirep"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@registry.register_task_model('embed', 'unirep')
class UniRepModel(UniRepAbstractModel):

    def __init__(self, config: UniRepConfig):
        super().__init__(config)
        self.embed_matrix = nn.Embedding(config.vocab_size, config.input_size)
        self.encoder = mLSTM(config)
        self.output_hidden_states = config.output_hidden_states
        self.init_weights()

    def forward(self, input_ids, input_mask=None):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        # fp16 compatibility
        input_mask = input_mask.to(dtype=next(self.parameters()).dtype)
        embedding_output = self.embed_matrix(input_ids)

        encoder_outputs = self.encoder(embedding_output, mask=input_mask)
        sequence_output = encoder_outputs[0]
        hidden_states = encoder_outputs[1]
        pooled_outputs = torch.cat(hidden_states, 1)

        outputs = (sequence_output, pooled_outputs)
        return outputs


@registry.register_task_model('language_modeling', 'unirep')
class UniRepForLM(UniRepAbstractModel):
    # TODO: Fix this for UniRep - UniRep changes the size of the targets

    def __init__(self, config):
        super().__init__(config)

        self.unirep = UniRepModel(config)
        self.feedforward = nn.Linear(config.hidden_size, config.vocab_size - 1)

        self.init_weights()

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None):

        outputs = self.unirep(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.feedforward(sequence_output)

        # add hidden states and if they are here
        outputs = (prediction_scores,) + outputs[2:]

        if targets is not None:
            targets = targets[:, 1:]
            prediction_scores = prediction_scores[:, :-1]
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), targets.view(-1))
            outputs = (lm_loss,) + outputs

        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('fluorescence', 'unirep')
@registry.register_task_model('stability', 'unirep')
class UniRepForValuePrediction(UniRepAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep = UniRepModel(config)
        self.predict = ValuePredictionHead(config.hidden_size * 2)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('remote_homology', 'unirep')
class UniRepForSequenceClassification(UniRepAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep = UniRepModel(config)
        self.classify = SequenceClassificationHead(
            config.hidden_size * 2, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('secondary_structure', 'unirep')
class UniRepForSequenceToSequenceClassification(UniRepAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep = UniRepModel(config)
        self.classify = SequenceToSequenceClassificationHead(
            config.hidden_size, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.classify(sequence_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states)
        return outputs


@registry.register_task_model('contact_prediction', 'unirep')
class UniRepForContactPrediction(UniRepAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.unirep = UniRepModel(config)
        self.predict = PairwiseContactPredictionHead(config.hidden_size, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, protein_length, input_mask=None, targets=None):

        outputs = self.unirep(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        outputs = self.predict(sequence_output, protein_length, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs
