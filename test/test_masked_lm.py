import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path  # noqa: E402
import math  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402
from tape_pytorch import models  # noqa: E402


torch.set_grad_enabled(False)
DATA_PATH = Path(__file__).parent.parent / 'data'
CONFIG_PATH = Path(__file__).parent.parent / 'config'


def run_dummy_pass(model, vocab_size, size):
    inputs = torch.randint(vocab_size, size).cuda()
    labels = torch.randint(vocab_size, size).cuda()
    attention_mask = torch.randint(2, size).cuda()

    loss, output, hidden = model(
        inputs, attention_mask=attention_mask, masked_lm_labels=labels)

    assert not math.isnan(loss.item())
    assert tuple(output.size()) == size + (vocab_size,)

    return (inputs, attention_mask, hidden)


def run_test_pass(model, inputs, attention_mask, hidden):
    sequence, pooled, hidden_all = model(
        inputs, attention_mask=attention_mask)
    assert torch.equal(sequence, hidden[-1])


def run_masked_lm_test(model_type, config):
    config.output_hidden_states = True
    base_model = model_type(config)
    model = models.MaskedLMModel(base_model, config)

    model.eval().cuda()

    result = run_dummy_pass(model, config.vocab_size, (4, 16))
    run_dummy_pass(model, config.vocab_size, (1, 1000))

    with tempfile.TemporaryDirectory() as tempdir:
        model.save_pretrained(tempdir)
        load_model = model_type.from_pretrained(tempdir).eval().cuda()
        run_test_pass(load_model, *result)


def test_transformer_masked_lm_model():
    config_file = CONFIG_PATH / 'bert_config.json'
    config = models.TAPEConfig.from_json_file(config_file)

    run_masked_lm_test(models.Transformer, config)
