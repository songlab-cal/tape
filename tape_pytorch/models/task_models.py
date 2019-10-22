import typing
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import BertOnlyMLMHead
from pytorch_transformers.modeling_bert import BertLayerNorm
from pytorch_transformers.modeling_utils import PretrainedConfig
from pytorch_transformers.modeling_utils import PreTrainedModel
from torch.nn.utils.weight_norm import weight_norm

from tape_pytorch.registry import registry

from .transformer import Transformer
from .resnet import ResNet
from .lstm import LSTM
from .unirep import Unirep


BASE_MODEL_CLASSES = {
    'transformer': Transformer,
    'resnet': ResNet,
    'lstm': LSTM,
    'unirep': Unirep}


class TAPEConfig(PretrainedConfig):
    r"""
        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
    """
    # pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 other_config_or_json_file,
                 base_model=None,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(other_config_or_json_file, str):
            with open(other_config_or_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(other_config_or_json_file, dict):
            for key, value in other_config_or_json_file.items():
                self.__dict__[key] = value
        elif isinstance(other_config_or_json_file, PretrainedConfig):
            for key, value in other_config_or_json_file.to_dict().items():
                self.__dict__[key] = value
        else:
            raise ValueError("First argument must be either a config file (PretrainedConfig)"
                             "or the path to a pretrained model config file (str)")

        if getattr(self, 'base_model', None) is None:
            if base_model is None:
                raise ValueError("Must pass a base model class")
            self.base_model = base_model

        if self.base_model not in BASE_MODEL_CLASSES:
            raise ValueError(f"Unirecognized base model class {self.base_model}")

        if not hasattr(self, 'hidden_act'):
            self.hidden_act = 'relu'

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(json_object)
        return config


class TAPEPreTrainedModel(PreTrainedModel):

    # Output keys
    SEQUENCE_EMBEDDING_KEY = 'sequence_embedding'
    POOLED_EMBEDDING_KEY = 'pooled_embedding'
    HIDDEN_STATES_KEY = 'hidden_states'
    ATTENTIONS_KEY = 'attentions'
    LOSS_KEY = 'loss'

    config_class = TAPEConfig
    base_model_prefix = "base_model"

    def _convert_outputs_to_dictionary(self, outputs: typing.Sequence[torch.Tensor]) \
            -> typing.Dict[str, torch.Tensor]:
        cls = self.__class__
        dict_outputs = {}
        dict_outputs[cls.SEQUENCE_EMBEDDING_KEY] = outputs[0]
        dict_outputs[cls.POOLED_EMBEDDING_KEY] = outputs[1]

        if self.config.output_hidden_states and self.config.output_attentions:
            dict_outputs[cls.HIDDEN_STATES_KEY] = outputs[2]
            dict_outputs[cls.ATTENTIONS_KEY] = outputs[3]
        elif self.config.output_hidden_states:
            dict_outputs[cls.HIDDEN_STATES_KEY] = outputs[2]
        elif self.config.output_attentions:
            dict_outputs[cls.ATTENTIONS_KEY] = outputs[2]

        return dict_outputs

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@registry.register_task_model('embed')
class EmbedModel(TAPEPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.base_model = BASE_MODEL_CLASSES[config.base_model](config)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask=None,
                target=None):
        # sequence_output, pooled_output, (hidden_states), (attention)
        outputs = self._convert_outputs_to_dictionary(
            self.base_model(input_ids, attention_mask=attention_mask))
        return outputs


@registry.register_task_model('pfam')
class MaskedLMModel(TAPEPreTrainedModel):

    TARGET_KEY = 'masked_lm_labels'
    PREDICTION_KEY = 'prediction_scores'
    PREDICTION_IS_SEQUENCE = True

    def __init__(self, config):
        super().__init__(config)
        self.base_model = BASE_MODEL_CLASSES[config.base_model](config)
        self.classify = BertOnlyMLMHead(config)

        if config.output_size != config.hidden_size:
            self.project = nn.Linear(config.output_size, config.hidden_size)
        else:
            self.project = lambda x: x

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.classify.predictions.decoder,
                                   self.base_model.embeddings.word_embeddings)

    def forward(self,
                input_ids,
                attention_mask=None,
                masked_lm_labels=None,
                clan_labels=None,
                family_labels=None):

        cls = self.__class__
        outputs = self._convert_outputs_to_dictionary(
            self.base_model(input_ids, position_ids=None, token_type_ids=None,
                            attention_mask=attention_mask, head_mask=None))
        prediction_scores = self.classify(self.project(outputs[cls.SEQUENCE_EMBEDDING_KEY]))

        outputs[cls.PREDICTION_KEY] = prediction_scores

        if masked_lm_labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
                ignore_index=-1)
            outputs[cls.LOSS_KEY] = masked_lm_loss

        return outputs


@registry.register_task_model('fluorescence')
@registry.register_task_model('stability')
class FloatPredictModel(TAPEPreTrainedModel):

    TARGET_KEY = 'target'
    PREDICTION_KEY = 'float_prediction'
    PREDICTION_IS_SEQUENCE = False

    def __init__(self, config):
        super().__init__(config)
        self.base_model = BASE_MODEL_CLASSES[config.base_model](config)
        self.predict = SimpleMLP(config.hidden_size, config.hidden_size * 2, 1, 0.5)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask=None,
                target=None):
        cls = self.__class__
        # sequence_output, pooled_output, (hidden_states), (attention)
        outputs = self._convert_outputs_to_dictionary(
            self.base_model(input_ids, attention_mask=attention_mask))
        float_prediction = self.predict(outputs[cls.POOLED_EMBEDDING_KEY])

        outputs[cls.PREDICTION_KEY] = float_prediction

        if target is not None:
            target = target.reshape_as(float_prediction)
            loss = F.mse_loss(float_prediction, target)
            outputs[cls.LOSS_KEY] = loss

        # (float_prediction_loss), float_prediction, (hidden_states), (attentions)
        return outputs


class SequenceClassificationModel(TAPEPreTrainedModel):

    TARGET_KEY = 'label'
    PREDICTION_KEY = 'class_scores'
    PREDICTION_IS_SEQUENCE = False

    def __init__(self, config, num_classes):
        super().__init__(config)
        self.base_model = BASE_MODEL_CLASSES[config.base_model](config)
        self.predict = SimpleMLP(
            config.hidden_size, config.hidden_size * 2, num_classes, 0.5)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask=None,
                label=None):
        cls = self.__class__
        # sequence_output, pooled_output, (hidden_states), (attention)
        outputs = self._convert_outputs_to_dictionary(
            self.base_model(input_ids, attention_mask=attention_mask))
        class_scores = self.predict(outputs[cls.POOLED_EMBEDDING_KEY])
        outputs[cls.PREDICTION_KEY] = class_scores

        if label is not None:
            loss = F.cross_entropy(class_scores, label)
            outputs[cls.LOSS_KEY] = loss

        return outputs  # (class_prediction_loss), class_scores, (hidden_states), (attentions)


@registry.register_task_model('remote_homology')
class RemoteHomologyModel(SequenceClassificationModel):

    def __init__(self, config):
        super().__init__(config, 1195)


class SequenceToSequenceClassificationModel(TAPEPreTrainedModel):

    TARGET_KEY = 'sequence_labels'
    PREDICTION_KEY = 'sequence_class_scores'
    PREDICTION_IS_SEQUENCE = True

    def __init__(self, config, num_classes: int):
        super().__init__(config)
        if config.num_classes is None:
            raise ValueError("Must pass value for num_classes")
        self.base_model = BASE_MODEL_CLASSES[config.base_model](config)
        self.predict = SimpleMLP(
            config.hidden_size, config.hidden_size * 2, num_classes, 0.5)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                attention_mask=None,
                sequence_labels=None,
                token_lengths=None):
        cls = self.__class__
        # sequence_output, pooled_output, (hidden_states), (attention)
        outputs = self._convert_outputs_to_dictionary(
            self.base_model(input_ids, attention_mask=attention_mask))

        sequence_embedding = outputs[cls.SEQUENCE_EMBEDDING_KEY]
        if token_lengths is not None:
            new_sequences = []
            for seq_embed, seq_tok_lengths in zip(sequence_embedding, token_lengths):
                expanded_seq = []
                for embed, n in zip(seq_embed, seq_tok_lengths):
                    if n == 0:
                        continue
                    embed = embed.repeat(n).view(n, self.config.hidden_size)
                    expanded_seq.append(embed)
                expanded_seq = torch.cat(expanded_seq, 0)
                new_sequences.append(expanded_seq)

            max_len = max(seq.size(0) for seq in new_sequences)
            new_sequences = [F.pad(embed, [0, 0, 0, max_len - embed.size(0)])
                             for embed in new_sequences]

            sequence_embedding = torch.stack(new_sequences, 0)
            sequence_embedding = sequence_embedding[:, :sequence_labels.size(1)]

        sequence_class_scores = self.predict(sequence_embedding)
        outputs[cls.PREDICTION_KEY] = sequence_class_scores

        if sequence_labels is not None:
            loss = F.cross_entropy(
                sequence_class_scores.view(-1, sequence_class_scores.size(2)),
                sequence_labels.view(-1),
                ignore_index=-1)
            outputs[cls.LOSS_KEY] = loss

        # (sequence_class_prediction_loss), class_scores, (hidden_states), (attentions)
        return outputs


@registry.register_task_model('secondary_structure')
class SS3ClassModel(SequenceToSequenceClassificationModel):

    def __init__(self, config):
        super().__init__(config, 3)


class SimpleMLP(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)
