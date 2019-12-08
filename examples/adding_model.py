"""Example of how to add a model in tape.

This file shows an example of how to add a new model to the tape training
pipeline. tape models follow the huggingface API and so require:

    - A config class
    - An abstract model class
    - A model class to output sequence and pooled embeddings
    - Task-specific classes for each individual task

This will walkthrough how to create each of these, with a task-specific class for
secondary structure prediction. You can look at the other task-specific classes
defined in e.g. tape/models/modeling_bert.py for examples on how to
define these other task-specific models for e.g. contact prediction or fluorescence
prediction.

In addition to defining these models, this shows how to register the model to
tape so that you can use the same training machinery to run your tasks.
"""


import torch
import torch.nn as nn
from tape import ProteinModel, ProteinConfig
from tape.models.modeling_utils import SequenceToSequenceClassificationHead
from tape.registry import registry


class SimpleConvConfig(ProteinConfig):
    """ The config class for our new model. This should be a subclass of
        ProteinConfig. It's a very straightforward definition, which just
        accepts the arguments that you would like the model to take in
        and assigns them to the class.

        Note - if you do not initialize using a model config file, you
        must provide defaults for all arguments.
    """

    def __init__(self,
                 vocab_size: int = 30,
                 filter_size: int = 128,
                 kernel_size: int = 5,
                 num_layers: int = 3,
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
            *[nn.Conv1d(config.filter_size, config.filter_size, config.kernel_size,
                        padding=config.kernel_size // 2)
              for _ in range(config.num_layers)])

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

    def forward(self, input_ids, input_mask=None, targets=None):
        """ Runs the forward model pass and may compute the loss if targets
            is present. Note that this does expect the third argument to be named
            `targets`. You can look at the different defined models to see
            what different tasks expect the label name to be.

        Args:
            input_ids (Tensor[long]):
                Tensor of input symbols of shape [batch_size x protein_length]
            input_mask (Tensor[bool]):
                Tensor of booleans w/ same shape as input_ids, indicating whether
                a given sequence position is valid
            targets (Tensor[long], optional):
                Tensor of output target labels of shape [batch_size x protein_length]
        """
        outputs = self.simple_conv(input_ids, input_mask)
        sequence_embedding = outputs[0]

        prediction = self.classify(sequence_embedding)

        outputs = (prediction,)

        if targets is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-1)(
                prediction.view(-1, prediction.size(2)), targets.view(-1))
            # cast to float b/c float16 does not have argmax support
            is_correct = prediction.float().argmax(-1) == targets
            is_valid_position = targets != -1

            # cast to float b/c otherwise torch does integer division
            num_correct = torch.sum(is_correct * is_valid_position).float()
            accuracy = num_correct / torch.sum(is_valid_position).float()
            metrics = {'acc': accuracy}

            outputs = ((loss, metrics),) + outputs

        return outputs  # ((loss, metrics)), prediction


if __name__ == '__main__':
    """ To actually run the model, you can do one of two things. You can
    simply import the appropriate run function from tape.main. The
    possible functions are `run_train`, `run_train_distributed`, `run_eval`,
    and `run_embed`. Alternatively, you can simply place this file inside
    the `tape/models` directory, where it will be auto-imported
    into tape.
    """
    from tape.main import run_train
    run_train()
