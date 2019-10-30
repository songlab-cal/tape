import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from pytorch_transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from pytorch_transformers.modeling_bert import BertLayerNorm

from tape_pytorch.registry import registry

from .task_models import TAPEPreTrainedModel, BASE_MODEL_CLASSES


class MaskedConv2d(nn.Conv2d):

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x)


def conv3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3 convolution with padding"""
    return MaskedConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1(in_planes, out_planes, stride=1):
    """1 convolution"""
    return MaskedConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, input_mask=None):
        identity = x

        out = self.conv1(x, input_mask)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, input_mask)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, input_mask=None):
        identity = x

        out = self.conv1(x, input_mask)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, input_mask)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, input_mask)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Downsample(nn.Module):
    def __init__(self, inplanes, planes, stride, norm_layer):
        super().__init__()
        self.conv1 = conv1(inplanes, planes, stride)
        self.norm_layer = norm_layer(planes)

    def forward(self, x, input_mask=None):
        x = self.conv1(x, input_mask)
        x = self.norm_layer(x)
        return x


class PairwiseFeatureExtractor(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError("out_features must be divisible by 2")
        self.downproject = nn.Linear(in_features, out_features // 2)

    def forward(self, inputs):
        inputs = self.downproject(inputs)
        seqlen = inputs.size(1)
        input_left = inputs.unsqueeze(2).repeat(1, 1, seqlen, 1)
        input_right = inputs.unsqueeze(1).repeat(1, seqlen, 1, 1)
        output = torch.cat([input_left, input_right], -1)

        return output


class SymmetricPredictor(nn.Module):

    def __init__(self, in_features: int):
        super().__init__()
        self.predict = nn.Linear(in_features, 2)

    def forward(self, inputs):
        prediction = self.predict(inputs)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        return prediction


class ResNetEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer

        config.cm_initial_hidden_dimension = 64
        config.cm_groups = 1
        config.cm_width_per_group = 64
        config.cm_block = 'bottleneck'
        config.cm_layers = (3, 4, 23, 3)
        config.cm_replace_stride_with_dilation = False

        block = {'basic': BasicBlock, 'bottleneck': Bottleneck}[config.cm_block]

        self.inplanes = config.cm_initial_hidden_dimension
        self.dilation = 1
        if config.cm_replace_stride_with_dilation in (False, None):
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            config.cm_replace_stride_with_dilation = [False, False, False]
        if len(config.cm_replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 config.cm_replace_stride_with_dilation))
        self.groups = config.cm_groups
        self.base_width = config.cm_width_per_group
        self.conv1 = MaskedConv2d(
            config.cm_initial_hidden_dimension, self.inplanes, kernel_size=7,
            stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, 2)
        # self.layer2 = self._make_layer(block, 16, config.cm_layers[1], stride=1,
                                       # dilate=config.cm_replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, 16, config.cm_layers[2], stride=1,
                                       # dilate=config.cm_replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 16, config.cm_layers[3], stride=1,
                                       # dilate=config.cm_replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample(self.inplanes, planes * block.expansion, stride, norm_layer)
            # downsample = nn.Sequential(
                # conv1(self.inplanes, planes * block.expansion, stride),
                # norm_layer(planes * block.expansion),
            # )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.ModuleList(layers)

    def run_function(self, start, chunk_size):
        def custom_forward(x, input_mask):
            for layer in self.layer1[start:start + chunk_size]:
                x = layer(x, input_mask=input_mask)
            return x
        return custom_forward

    def forward(self, x, input_mask=None, chunks=None):
        x = x.permute(0, 3, 1, 2)
        if input_mask is not None:
            input_mask = input_mask.permute(0, 3, 1, 2)
        x = self.conv1(x, input_mask)
        x = self.bn1(x)
        x = self.relu(x)
        if chunks is not None:
            assert isinstance(chunks, int)
            chunk_size = (len(self.layer1) + chunks - 1) // chunks
            for start in range(0, len(self.layer1), chunk_size):
                x = checkpoint(self.run_function(start, chunk_size), x, input_mask)
        else:
            for module in self.layer1:
                x = module(x, input_mask)
        # for module in self.layer2:
            # x = module(x, input_mask)
        # for module in self.layer3:
            # x = module(x, input_mask)
        # for module in self.layer4:
            # x = module(x, input_mask)
        x = x.permute(0, 2, 3, 1)
        return x


@registry.register_task_model('contact_prediction')
class ContactPredictor(TAPEPreTrainedModel):

    TARGET_KEY = 'true_contacts'
    PREDICTION_KEY = 'contact_scores'
    PREDICTION_IS_SEQUENCE = True

    def __init__(self, config):
        super().__init__(config)
        self.base_model = BASE_MODEL_CLASSES[config.base_model](config)
        self.encoder = ResNetEncoder(config)
        self.feature_extractor = PairwiseFeatureExtractor(config.output_size, 64)
        self.predict = SymmetricPredictor(64)

    def forward(self,
                input_ids,
                attention_mask=None,
                contact_labels=None):
        cls = self.__class__

        outputs = self._convert_outputs_to_dictionary(
            self.base_model(input_ids, attention_mask=attention_mask))
        sequence_embedding = outputs[cls.SEQUENCE_EMBEDDING_KEY]

        pairwise_features = checkpoint(self.feature_extractor, sequence_embedding)
        # pairwise_features = self.feature_extractor(sequence_embedding)

        if attention_mask is not None:
            pairwise_mask = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
            pairwise_mask = pairwise_mask.unsqueeze(3).type_as(pairwise_features)
        else:
            pairwise_mask = None

        encoded_output = self.encoder(pairwise_features, input_mask=pairwise_mask, chunks=1)
        prediction = self.predict(encoded_output)

        outputs[cls.PREDICTION_KEY] = prediction

        if contact_labels is not None:
            loss = F.cross_entropy(
                prediction[:, 1:-1, 1:-1].contiguous().view(-1, 2),
                contact_labels.view(-1),
                ignore_index=-1)
            outputs[cls.LOSS_KEY] = loss

        return outputs
