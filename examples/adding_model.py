import torch
import torch.nn as nn
import torch.nn.functional as F
from protein_models import ProteinConfig
from protein_models import ProteinModel
from protein_models.modeling_utils import SequenceToSequenceClassificationHead
from tape_pytorch.registry import registry


class SimpleConvConfig(ProteinConfig):
    """ The config class for our new model. This should be a subclass of
        ProteinConfig. It's a very straightforward definition, which just
        accepts the arguments that you would like the model to take in
        and assigns them to the class.
    """

    def __init__(self,
                 vocab_size: int,
                 filter_size: int,
                 kernel_size: int,
                 num_layers: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers


class SimpleConvAbstractModel(ProteinModel):
    """ All your models will inherit from this one - it's used to define the
        config_class of the model set and also to define the base_model_prefix.
        This is used to allow easy loading/saving into different models.
    """
    config_class = SimpleConvConfig
    base_model_prefix = 'simple_conv'


class SimpleConvModel(SimpleConvAbstractModel):
    """ The base model class. This will return embeddings of the input amino
        acid sequence. It is not used for any specific task - you'll have to
        define task-specific models further on. Note that there is a little
        more machinery in the models we define, but this is a stripped down
        version that should give you what you need
    """
    # init expects only a single argument - the config
    def __init__(self, config: SimpleConvConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.filter_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(config.filter_size, config.filter_size, config.kernel_size,
                      padding=config.kernel_size // 2)
            for _ in range(config.num_layers))

        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids, input_mask=None):
        """ Runs the forward model pass

        Args:
            input_ids (Tensor[long]):
                Tensor of input symbols of shape [batch_size x protein_length]
            input_mask (Tensor[bool]):
                Tensor of booleans w/ same shape as input_ids, indicating whether
                a given sequence position is valid

        Returns:
            sequence_embedding (Tensor[float]):
                Embedded sequence of shape [batch_size x protein_length x hidden_size]
            pooled_embedding (Tensor[float]):
                Pooled representation of the entire sequence of size [batch_size x hidden_size]
        """

        # Embed the input_ids
        embed = self.embedding(input_ids)

        # Pass embeddings through the encoder - you may want to use
        # the input mask here (not used in this example, but is generally
        # used in most of our models).
        embed = embed.permute(0, 2, 1)  # Conv layers are NCW
        sequence_embedding = self.encoder(embed)

        # Compute the pooled embedding - you can do arbitrarily complicated
        # things to do this, here we're just going to mean-pool the result
        pooled_embedding = self.pooler(sequence_embedding).squeeze(2)

        # Re-permute the sequence embedding to be NWC
        sequence_embedding = sequence_embedding.permute(0, 2, 1).contiguous()

        outputs = (sequence_embedding, pooled_embedding)
        return outputs


# This registers the model to a specific task, allowing you to use all of TAPE's
# machinery to train it.
@registry.register_task_model('secondary_structure', 'simple-conv')
class SimpleConvForSequenceToSequenceClassification(SimpleConvAbstractModel):

    def __init__(self, config: SimpleConvConfig):
        super().__init__(config)
        # the name of this variable *must* match the base_model_prefix
        self.simple_conv = SimpleConvModel(config)
        # The seq2seq classification head. First argument must match the
        # output embedding size of the SimpleConvModel. The second argument
        # is present in every config (it's an argument of ProteinConfig)
        # and is used for classification tasks.
        self.classify = SequenceToSequenceClassificationHead(
            config.filter_size, config.num_labels)

    def forward(self, input_ids, input_mask=None, amino_acid_labels=None):
        # TODO (roshan): Standardize label names
        """ Runs the forward model pass and may compute the loss if amino_acid_labels
            is present. Note that this does expect the third argument to be named
            `amino_acid_labels`. You can look at the different defined models to see
            what different tasks expect the label name to be.

        Args:
            input_ids (Tensor[long]):
                Tensor of input symbols of shape [batch_size x protein_length]
            input_mask (Tensor[bool]):
                Tensor of booleans w/ same shape as input_ids, indicating whether
                a given sequence position is valid

        Returns:
            sequence_embedding (Tensor[float]):
                Embedded sequence of shape [batch_size x protein_length x hidden_size]
            pooled_embedding (Tensor[float]):
                Pooled representation of the entire sequence of size [batch_size x hidden_size]
        """
        outputs = self.simple_conv(input_ids, input_mask)
        sequence_embedding = outputs[0]

        prediction = self.classify(sequence_embedding)

        outputs = (prediction,)

        if amino_acid_labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-1)(
                prediction.view(-1, prediction.size(2)), amino_acid_labels.view(-1))
            # cast to float b/c float16 does not have argmax support
            is_correct = prediction.float().argmax(-1) == amino_acid_labels
            is_valid_position = amino_acid_labels != -1

            accuracy = torch.sum(is_correct * is_valid_position) / torch.sum(is_valid_position)
            metrics = {'acc': accuracy}

            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction


if __name__ == '__main__':
    from tape_pytorch.main import run_train
    run_train()
