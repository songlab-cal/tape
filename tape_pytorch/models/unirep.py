import json
import torch
import torch.nn as nn
from torch.nn.modules.rnn import RNNBase, LSTMCell

from pytorch_transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from pytorch_transformers.modeling_bert import BertEmbeddings

from tape_pytorch.registry import registry


class UnirepConfig(PretrainedConfig):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the
        configuration of a `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
    """
    # pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=8000,
                 input_size: int = 10,
                 hidden_size: int = 1900,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=8096,
                 layer_norm_eps=1e-12,
                 initializer_range=0.02,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.layer_norm_eps = layer_norm_eps
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

        self.type_vocab_size = 1
        self.output_size = 2 * self.hidden_size


class UnirepPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size, config.output_size)
        self.activation = nn.Tanh()
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states):
        h_out, c_out = hidden_states

        # Permute things around
        # h_out = h_out.view(self.num_hidden_layers, 2, -1, self.hidden_size)
        h_out = h_out.transpose(1, 0).reshape(-1, 2 * self.num_hidden_layers * self.hidden_size)
        x = self.dense(h_out)
        x = self.activation(x)
        return x


# TODO: Work with pack_padded_sequence
class mLSTM(RNNBase):

    def __init__(self, config):
        super().__init__(mode='LSTM', input_size=config.input_size,
                         hidden_size=config.hidden_size,
                         num_layers=1, bias=config.use_bias, batch_first=True,
                         dropout=config.dropout, bidirectional=False)
        self.input_project = nn.Linear(config.input_size, config.hidden_size)
        self.hidden_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.lstm_cell = LSTMCell(config.input_size, config.hidden_size, config.use_bias)

        self.hidden_size = config.hidden_size

    def forward(self, inputs, hx=None, mask=None):
        batch_size = inputs.size(0)
        seqlen = inputs.size(1)

        if hx is None:
            zeros = torch.zeros(batch_size, self.hidden_size,
                                dtype=inputs.dtype, device=inputs.device)
            hx = (zeros, zeros)

        hx, cx = hx

        steps = []
        for seq in range(seqlen):
            prev = (hx, cx)
            seq_input = inputs[:, seq, :]
            mx = self.input_project(seq_input) * self.hidden_project(hx)
            hx = (mx, cx)
            hx, cx = self.lstm_cell(seq_input, hx)
            if mask is not None:
                seqmask = mask[:, seq]
                hx = seqmask * hx + (1 - seqmask) * prev[0]
                cx = seqmask * cx + (1 - seqmask) * prev[1]
            steps.append(cx)

        return torch.stack(steps, 1), (hx, cx)


@registry.register_model('unirep')
class Unirep(PreTrainedModel):
    config_class = UnirepConfig

    def __init__(self, config: UnirepConfig):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = nn.LSTM(
            config.hidden_size, config.hidden_size, config.num_hidden_layers,
            batch_first=True, bidirectional=True, dropout=config.hidden_dropout_prob)
        self.encoder.flatten_parameters()
        self.pooler = UnirepPooler(config)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # extended_attention_mask = attention_mask.unsqueeze(2)
        # fp16 compatibility
        # extended_attention_mask = extended_attention_mask.to(
            # dtype=next(self.parameters()).dtype)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        sequence_output, hidden_states = self.encoder(embedding_output)
        pooled_output = self.pooler(hidden_states)

        outputs = (sequence_output, pooled_output,)
        return outputs
