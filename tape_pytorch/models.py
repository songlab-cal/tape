import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import BertConfig
from pytorch_transformers.modeling_bert import BertOnlyMLMHead
from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.modeling_bert import BertLayerNorm
from pytorch_transformers.modeling_utils import PreTrainedModel
from torch.nn.utils.weight_norm import weight_norm


TAPEConfig = BertConfig
Transformer = BertModel


class ResNet(nn.Module):
    pass


class LSTM(nn.Module):
    pass


class UniRep(nn.Module):
    pass


class Bepler(nn.Module):
    pass


class MaskedLMModel(PreTrainedModel):

    def __init__(self, base_model, config):
        super().__init__(config)

        self.bert = base_model
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()
        self.tie_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
                ignore_index=-1)
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class FloatPredictModel(nn.Module):

    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.predict = nn.Linear(config.hidden_size, config.hidden_size * 2, 1)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                label=None):
        outputs = self.base_model(  # sequence_output, pooled_output, (hidden_states), (attention)
            input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        float_prediction = self.predict(pooled_output)

        outputs = (float_prediction,) + outputs[2:]  # Add hidden states and attention if they are here

        if label is not None:
            loss = F.mse_loss(float_prediction, label)
            outputs = (loss,) + outputs

        return outputs  # (float_prediction_loss), float_prediction, (hidden_states), (attentions)


class SequenceClassificationModel(nn.Module):

    def __init__(self, base_model, config, num_classes: int):
        self.base_model = base_model
        self.predict = nn.Linear(config.hidden_size, config.hidden_size * 2, num_classes)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                label=None):
        outputs = self.base_model(  # sequence_output, pooled_output, (hidden_states), (attention)
            input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        class_scores = self.predict(pooled_output)

        outputs = (class_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if label is not None:
            loss = F.cross_entropy(class_scores, label)
            outputs = (loss,) + outputs

        return outputs  # (class_prediction_loss), class_scores, (hidden_states), (attentions)


class SequenceToSequenceClassificationModel(nn.Module):

    def __init__(self, base_model, config, num_classes: int):
        self.base_model = base_model
        self.predict = nn.Linear(config.hidden_size, config.hidden_size * 2, num_classes)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                label=None):
        outputs = self.base_model(  # sequence_output, pooled_output, (hidden_states), (attention)
            input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_class_scores = self.predict(sequence_output)

        outputs = (sequence_class_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if label is not None:
            loss = F.cross_entropy(
                sequence_class_scores.view(-1, sequence_class_scores.size(2)),
                label.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (sequence_class_prediction_loss), class_scores, (hidden_states), (attentions)


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
