from tape_pytorch.datasets import PfamDataset
from pathlib import Path
from tqdm import tqdm


def test_pfam():
    dataset = PfamDataset(Path(__file__).parent.parent / 'data', 'train')

    for _ in tqdm(dataset):
        pass
