"""Code for implementing protein model from Bepler & Berger (ICLR 2019).

Note that this is *not* integrated into tape-pytorch at the modment, due large differences in
the way the model is constructed/trained relative to the other models. See the original repo
at https://github.com/tbepler/protein-sequence-embedding-iclr2019.git for full training code.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence

from pytorch_transformers.modeling_utils import PretrainedConfig


class BeplerConfig(PretrainedConfig):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the
        configuration of a `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
    """
    # pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=8000,
                 num_hidden_layers: int = 3,
                 hidden_size: int = 1024,
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
            self.num_hidden_layers = num_hidden_layers
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


class LSTMPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.num_hidden_layers * config.output_size, config.output_size)
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


class LMEmbed(nn.Module):
    def __init__(self, nin, nout, lm, padding_idx=-1, transform=nn.ReLU(), sparse=False):
        super(LMEmbed, self).__init__()

        if padding_idx == -1:
            padding_idx = nin - 1

        self.lm = lm
        self.embed = nn.Embedding(nin, nout, padding_idx=padding_idx, sparse=sparse)
        self.proj = nn.Linear(lm.hidden_size(), nout)
        self.transform = transform
        self.nout = nout

    def forward(self, x):
        packed = type(x) is PackedSequence
        h_lm = self.lm.encode(x)

        # embed and unpack if packed
        if packed:
            h = self.embed(x.data)
            h_lm = h_lm.data
        else:
            h = self.embed(x)

        # project
        h_lm = self.proj(h_lm)
        h = self.transform(h + h_lm)

        # repack if needed
        if packed:
            h = PackedSequence(h, x.batch_sizes)

        return h


class Linear(nn.Module):
    def __init__(self, nin, nhidden, nout, padding_idx=-1,
                 sparse=False, lm=None):
        super(Linear, self).__init__()

        if padding_idx == -1:
            padding_idx = nin - 1

        if lm is not None:
            self.embed = LMEmbed(nin, nhidden, lm, padding_idx=padding_idx, sparse=sparse)
            self.proj = nn.Linear(self.embed.nout, nout)
            self.lm = True
        else:
            self.proj = nn.Embedding(nin, nout, padding_idx=padding_idx, sparse=sparse)
            self.lm = False

        self.nout = nout

    def forward(self, x):
        if self.lm:
            h = self.embed(x)
            if type(h) is PackedSequence:
                h = h.data
                z = self.proj(h)
                z = PackedSequence(z, x.batch_sizes)
            else:
                h = h.view(-1, h.size(2))
                z = self.proj(h)
                z = z.view(x.size(0), x.size(1), -1)
        else:
            if type(x) is PackedSequence:
                z = self.embed(x.data)
                z = PackedSequence(z, x.batch_sizes)
            else:
                z = self.embed(x)

        return z


class StackedRNN(nn.Module):
    def __init__(self, nin, nembed, nunits, nout, nlayers=2, padding_idx=-1, dropout=0,
                 rnn_type='lstm', sparse=False, lm=None):
        super(StackedRNN, self).__init__()

        if padding_idx == -1:
            padding_idx = nin - 1

        if lm is not None:
            self.embed = LMEmbed(nin, nembed, lm, padding_idx=padding_idx, sparse=sparse)
            nembed = self.embed.nout
            self.lm = True
        else:
            self.embed = nn.Embedding(nin, nembed, padding_idx=padding_idx, sparse=sparse)
            self.lm = False

        if rnn_type == 'lstm':
            RNN = nn.LSTM
        elif rnn_type == 'gru':
            RNN = nn.GRU

        self.dropout = nn.Dropout(p=dropout)
        if nlayers == 1:
            dropout = 0

        self.rnn = RNN(nembed, nunits, nlayers, batch_first=True,
                       bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(2 * nunits, nout)
        self.nout = nout

    def forward(self, x):
        if self.lm:
            h = self.embed(x)
        else:
            if type(x) is PackedSequence:
                h = self.embed(x.data)
                h = PackedSequence(h, x.batch_sizes)
            else:
                h = self.embed(x)
        h, _ = self.rnn(h)

        if type(h) is PackedSequence:
            h = h.data
            h = self.dropout(h)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = h.view(-1, h.size(2))
            h = self.dropout(h)
            z = self.proj(h)
            z = z.view(x.size(0), x.size(1), -1)

        return z


class BiLM(nn.Module):
    def __init__(self, nin, nout, embedding_dim, hidden_dim, num_layers,
                 tied=True, mask_idx=None, dropout=0):
        super(BiLM, self).__init__()

        if mask_idx is None:
            mask_idx = nin - 1
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(nin, embedding_dim, padding_idx=mask_idx)
        self.dropout = nn.Dropout(p=dropout)

        self.tied = tied
        if tied:
            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.rnn = nn.ModuleList(layers)
        else:
            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.lrnn = nn.ModuleList(layers)

            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.rrnn = nn.ModuleList(layers)

        self.linear = nn.Linear(hidden_dim, nout)

    def hidden_size(self):
        h = 0
        if self.tied:
            for layer in self.rnn:
                h += 2 * layer.hidden_size
        else:
            for layer in self.lrnn:
                h += layer.hidden_size
            for layer in self.rrnn:
                h += layer.hidden_size
        return h

    def reverse(self, h):
        packed = type(h) is PackedSequence
        if packed:
            h, batch_sizes = pad_packed_sequence(h, batch_first=True)
            h_rvs = h.clone().zero_()
            for i in range(h.size(0)):
                n = batch_sizes[i]
                idx = [j for j in range(n - 1, -1, -1)]
                idx = torch.LongTensor(idx).to(h.device)
                h_rvs[i, :n] = h[i].index_select(0, idx)
            # repack h_rvs
            h_rvs = pack_padded_sequence(h_rvs, batch_sizes, batch_first=True)
        else:
            idx = [i for i in range(h.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx).to(h.device)
            h_rvs = h.index_select(1, idx)
        return h_rvs

    def transform(self, z_fwd, z_rvs, last_only=False):
        # sequences are flanked by the start/stop token as:
        # [stop, x, stop]

        # z_fwd should be [stop,x]
        # z_rvs should be [x,stop] reversed

        # first, do the forward direction
        if self.tied:
            layers = self.rnn
        else:
            layers = self.lrnn

        h_fwd = []
        h = z_fwd
        for rnn in layers:
            h, _ = rnn(h)
            if type(h) is PackedSequence:
                h = PackedSequence(self.dropout(h.data), h.batch_sizes)
            else:
                h = self.dropout(h)
            if not last_only:
                h_fwd.append(h)
        if last_only:
            h_fwd = h

        # now, do the reverse direction
        if self.tied:
            layers = self.rnn
        else:
            layers = self.rrnn

        # we'll need to reverse the direction of these
        # hidden states back to match forward direction

        h_rvs = []
        h = z_rvs
        for rnn in layers:
            h, _ = rnn(h)
            if type(h) is PackedSequence:
                h = PackedSequence(self.dropout(h.data), h.batch_sizes)
            else:
                h = self.dropout(h)
            if not last_only:
                h_rvs.append(self.reverse(h))
        if last_only:
            h_rvs = self.reverse(h)

        return h_fwd, h_rvs

    def embed_and_split(self, x, pad=False):
        packed = type(x) is PackedSequence
        if packed:
            x, batch_sizes = pad_packed_sequence(x, batch_first=True)

        if pad:
            # pad x with the start/stop token
            x = x + 1
            # append start/stop tokens to x
            x_ = x.data.new(x.size(0), x.size(1) + 2).zero_()
            if packed:
                for i in range(len(batch_sizes)):
                    n = batch_sizes[i]
                    x_[i, 1:n + 1] = x[i, :n]
                batch_sizes = [s + 2 for s in batch_sizes]
            else:
                x_[:, 1:-1] = x
            x = x_

        # sequences x are flanked by the start/stop token as:
        # [stop, x, stop]

        # now, encode x as distributed vectors
        z = self.embed(x)

        # to pass to transform, we discard the last element for
        # z_fwd and the first element for z_rvs
        z_fwd = z[:, :-1]
        z_rvs = z[:, 1:]
        if packed:
            lengths = [s - 1 for s in batch_sizes]
            z_fwd = pack_padded_sequence(z_fwd, lengths, batch_first=True)
            z_rvs = pack_padded_sequence(z_rvs, lengths, batch_first=True)
        # reverse z_rvs
        z_rvs = self.reverse(z_rvs)

        return z_fwd, z_rvs

    def encode(self, x):
        z_fwd, z_rvs = self.embed_and_split(x, pad=True)
        h_fwd_layers, h_rvs_layers = self.transform(z_fwd, z_rvs)

        # concatenate hidden layers together
        packed = type(z_fwd) is PackedSequence
        concat = []
        for h_fwd, h_rvs in zip(h_fwd_layers, h_rvs_layers):
            if packed:
                h_fwd, batch_sizes = pad_packed_sequence(h_fwd, batch_first=True)
                h_rvs, batch_sizes = pad_packed_sequence(h_rvs, batch_first=True)
            # discard last element of h_fwd and first element of h_rvs
            h_fwd = h_fwd[:, :-1]
            h_rvs = h_rvs[:, 1:]

            # accumulate for concatenation
            concat.append(h_fwd)
            concat.append(h_rvs)

        h = torch.cat(concat, 2)
        if packed:
            batch_sizes = [s - 1 for s in batch_sizes]
            h = pack_padded_sequence(h, batch_sizes, batch_first=True)

        return h

    def forward(self, x):
        # x's are already flanked by the star/stop token as:
        # [stop, x, stop]
        z_fwd, z_rvs = self.embed_and_split(x, pad=False)

        h_fwd, h_rvs = self.transform(z_fwd, z_rvs, last_only=True)

        packed = type(z_fwd) is PackedSequence
        if packed:
            h_flat = h_fwd.data
            logp_fwd = self.linear(h_flat)
            logp_fwd = PackedSequence(logp_fwd, h_fwd.batch_sizes)

            h_flat = h_rvs.data
            logp_rvs = self.linear(h_flat)
            logp_rvs = PackedSequence(logp_rvs, h_rvs.batch_sizes)

            logp_fwd, batch_sizes = pad_packed_sequence(logp_fwd, batch_first=True)
            logp_rvs, batch_sizes = pad_packed_sequence(logp_rvs, batch_first=True)

        else:
            b = h_fwd.size(0)
            n = h_fwd.size(1)
            h_flat = h_fwd.contiguous().view(-1, h_fwd.size(2))
            logp_fwd = self.linear(h_flat)
            logp_fwd = logp_fwd.view(b, n, -1)

            h_flat = h_rvs.contiguous().view(-1, h_rvs.size(2))
            logp_rvs = self.linear(h_flat)
            logp_rvs = logp_rvs.view(b, n, -1)

        # prepend forward logp with zero
        # postpend reverse logp with zero

        b = h_fwd.size(0)
        zero = h_fwd.data.new(b, 1, logp_fwd.size(2)).zero_()
        logp_fwd = torch.cat([zero, logp_fwd], 1)
        logp_rvs = torch.cat([logp_rvs, zero], 1)

        logp = F.log_softmax(logp_fwd + logp_rvs, dim=2)

        if packed:
            batch_sizes = [s + 1 for s in batch_sizes]
            logp = pack_padded_sequence(logp, batch_sizes, batch_first=True)

        return logp
