import json
import torch
import torch.nn as nn

from pytorch_transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from pytorch_transformers.modeling_bert import BertLayerNorm

from tape_pytorch.registry import registry


class MaskedConv1d(nn.Conv1d):

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x)


class TransposeLayerNorm(BertLayerNorm):

    def forward(self, x, input_mask=None):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ResNetConfig(PretrainedConfig):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the
        configuration of a `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
    """
    # pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=8000,
                 block='bottleneck',
                 layers=(3, 4, 23, 3),
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=False,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=8096,
                 initial_hidden_dimension=64,
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
            self.block = block
            self.layers = layers
            self.zero_init_residual = zero_init_residual
            self.groups = groups
            self.width_per_group = width_per_group
            self.replace_stride_with_dilation = replace_stride_with_dilation
            self.hidden_dropout_prob = hidden_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initial_hidden_dimension = initial_hidden_dimension
            self.layer_norm_eps = layer_norm_eps
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

        self.hidden_size = self.initial_hidden_dimension
        self.output_size = 2048
        self.type_vocab_size = 1


def conv3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3 convolution with padding"""
    return MaskedConv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1(in_planes, out_planes, stride=1):
    """1 convolution"""
    return MaskedConv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = TransposeLayerNorm
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
            norm_layer = TransposeLayerNorm
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


class ResNetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(ResNetEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.initial_hidden_dimension, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.initial_hidden_dimension)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.initial_hidden_dimension)

        # self.LayerNorm is not snake-cased to stick with TensorFlow
        # model variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(
            config.initial_hidden_dimension, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ResNetPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.output_size, config.output_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ResNetEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        norm_layer = TransposeLayerNorm
        self._norm_layer = norm_layer

        block = {'basic': BasicBlock, 'bottleneck': Bottleneck}[config.block]

        self.inplanes = config.initial_hidden_dimension
        self.dilation = 1
        if config.replace_stride_with_dilation in (False, None):
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            config.replace_stride_with_dilation = [False, False, False]
        if len(config.replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 config.replace_stride_with_dilation))
        self.groups = config.groups
        self.base_width = config.width_per_group
        self.conv1 = MaskedConv1d(
            config.initial_hidden_dimension, self.inplanes, kernel_size=7,
            stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, config.layers[0])
        self.layer2 = self._make_layer(block, 128, config.layers[1], stride=1,
                                       dilate=config.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, config.layers[2], stride=1,
                                       dilate=config.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, config.layers[3], stride=1,
                                       dilate=config.replace_stride_with_dilation[2])

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

    def forward(self, x, input_mask=None):
        x = self.conv1(x, input_mask)
        x = self.bn1(x)
        x = self.relu(x)
        for module in self.layer1:
            x = module(x, input_mask)
        for module in self.layer2:
            x = module(x, input_mask)
        for module in self.layer3:
            x = module(x, input_mask)
        for module in self.layer4:
            x = module(x, input_mask)

        return x


@registry.register_model('resnet')
class ResNet(PreTrainedModel):
    config_class = ResNetConfig

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = ResNetEmbeddings(config)
        self.encoder = ResNetEncoder(config)

        self.pooler = ResNetPooler(config)

        self.apply(self.init_weights)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like
        # an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if config.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal
            # for initialization cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(2)
        # fp16 compatibility
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        sequence_output = self.encoder(embedding_output.transpose(1, 2),
                                       extended_attention_mask.transpose(1, 2)).transpose(1, 2)
        # sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,)
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
