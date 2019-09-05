import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertForMaskedLM
from torch.nn.utils.weight_norm import weight_norm


Transformer = BertForMaskedLM


class ResNet(nn.Module):
    pass


class LSTM(nn.Module):
    pass


class UniRep(nn.Module):
    pass


class Bepler(nn.Module):
    pass


class FloatPredictModel(nn.Module):

    def __init__(self, base_model, config):
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
