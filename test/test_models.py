from pathlib import Path
import math
import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch  # noqa: E402
from tape import models  # noqa: E402


DATA_PATH = Path(__file__).parent.parent / 'data'
CONFIG_PATH = Path(__file__).parent.parent / 'config'

config_file = CONFIG_PATH / 'bert_config.json'
config = models.TAPEConfig.from_json_file(config_file)

torch.set_grad_enabled(False)
model = models.Transformer(config)
model.eval().cuda()


def test_floatpredict_model():
    task_model = models.FloatPredictModel(model, config)
    task_model.eval().cuda()

    def run_dummy_pass(*size):
        inputs = torch.randint(config.vocab_size, size).cuda()
        attention_mask = torch.randint(2, size).cuda()
        labels = torch.randn(size[0], 1).cuda()

        loss, output = task_model(
            inputs, attention_mask=attention_mask, label=labels)

        assert not math.isnan(loss.item())
        assert tuple(output.size()) == (size[0], 1)

    run_dummy_pass(4, 16)
    run_dummy_pass(1, 1000)


def test_classpredict_model():
    num_classes = random.randint(15, 128)
    task_model = models.SequenceClassificationModel(model, config, num_classes)
    task_model.eval().cuda()

    def run_dummy_pass(*size):
        inputs = torch.randint(config.vocab_size, size).cuda()
        attention_mask = torch.randint(2, size).cuda()
        labels = torch.randint(num_classes, (size[0],)).cuda()

        loss, output = task_model(
            inputs, attention_mask=attention_mask, label=labels)

        assert not math.isnan(loss.item())
        assert tuple(output.size()) == (size[0], num_classes)

    run_dummy_pass(4, 16)
    run_dummy_pass(1, 1000)


def test_seqpredict_model():
    num_classes = random.randint(15, 128)
    task_model = models.SequenceToSequenceClassificationModel(model, config, num_classes)
    task_model.eval().cuda()

    def run_dummy_pass(*size):
        inputs = torch.randint(config.vocab_size, size).cuda()
        attention_mask = torch.randint(2, size).cuda()
        labels = torch.randint(num_classes, size).cuda()

        loss, output = task_model(
            inputs, attention_mask=attention_mask, label=labels)

        assert not math.isnan(loss.item())
        assert tuple(output.size()) == size + (num_classes,)

    run_dummy_pass(4, 16)
    run_dummy_pass(1, 1000)
