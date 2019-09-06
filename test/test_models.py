import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path  # noqa: E402
import math  # noqa: E402

import torch  # noqa: E402
from tape_pytorch import models  # noqa: E402


DATA_PATH = Path(__file__).parent.parent / 'data'
CONFIG_PATH = Path(__file__).parent.parent / 'config'

config_file = CONFIG_PATH / 'bert_config.json'
config = models.TAPEConfig.from_json_file(config_file)

model = models.Transformer(config)
model.cuda()


def test_floatpredict_model():
    task_model = models.FloatPredictModel(model, config)
    task_model.cuda()

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
