from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch  # noqa: E402
from tape_pytorch.models.resnet import ResNet, ResNetConfig  # noqa: E402


DATA_PATH = Path(__file__).parent.parent / 'data'
CONFIG_PATH = Path(__file__).parent.parent / 'config'


torch.set_grad_enabled(False)


def test_resnet():
    config = ResNetConfig()
    model = ResNet(config).eval().cuda()

    def run_dummy_pass(*size):
        inputs = torch.randint(config.vocab_size, size).cuda()
        attention_mask = torch.randint(2, size).cuda()

        seq_out, pool_out = model(
            inputs, attention_mask=attention_mask)

        assert seq_out.shape == size + (2048,)
        assert pool_out.shape == (size[0], 2048)

        inputs[attention_mask == 0] = 0
        seq_out1, pool_out1 = model(
            inputs, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(2).to(seq_out.dtype)
        masked_out1 = seq_out1 * mask
        masked_out = seq_out * mask
        assert torch.equal(masked_out, masked_out1)

    run_dummy_pass(4, 16)
    run_dummy_pass(1, 1000)
