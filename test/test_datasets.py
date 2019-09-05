from tape_pytorch import datasets
from pathlib import Path
import numpy as np
import random

DATA_PATH = Path(__file__).parent.parent / 'data'


def test_pfam_dataset():
    dataset = datasets.PfamDataset(DATA_PATH, 'train')

    # Test 10 random accesses
    for _ in range(10):
        masked_ids, attention_mask, labels, clan, family = dataset[random.randint(0, len(dataset) - 1)]
        assert isinstance(masked_ids, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(clan, int)
        assert isinstance(family, int)

        assert 0 <= masked_ids.min() <= masked_ids.max() < dataset.tokenizer.vocab_size
        assert np.all((attention_mask == 0) | (attention_mask == 1))
        assert -1 <= labels.min() <= labels.max() < dataset.tokenizer.vocab_size


def test_secondary_structure_dataset():
    dataset = datasets.SecondaryStructureDataset(DATA_PATH, 'train', num_classes=3)

    # Test 10 random accesses
    for _ in range(10):
        ids, attention_mask, structure_labels = dataset[random.randint(0, len(dataset) - 1)]
        assert isinstance(ids, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert isinstance(structure_labels, np.ndarray)

        assert 0 <= ids.min() <= ids.max() < dataset.tokenizer.vocab_size
        assert np.all((attention_mask == 0) | (attention_mask == 1))
        assert 0 <= structure_labels.min() <= structure_labels.max() < 3

    dataset = datasets.SecondaryStructureDataset(DATA_PATH, 'train', num_classes=8)

    # Test 10 random accesses
    for _ in range(10):
        ids, attention_mask, structure_labels = dataset[random.randint(0, len(dataset) - 1)]
        assert isinstance(ids, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert isinstance(structure_labels, np.ndarray)

        assert 0 <= ids.min() <= ids.max() < dataset.tokenizer.vocab_size
        assert np.all((attention_mask == 0) | (attention_mask == 1))
        assert 0 <= structure_labels.min() <= structure_labels.max() < 8


def test_remote_homology_dataset():
    dataset = datasets.RemoteHomologyDataset(DATA_PATH, 'train')

    # Test 10 random accesses
    for _ in range(10):
        ids, attention_mask, fold_label = dataset[random.randint(0, len(dataset) - 1)]
        assert isinstance(ids, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert isinstance(fold_label, int)

        assert 0 <= ids.min() <= ids.max() < dataset.tokenizer.vocab_size
        assert np.all((attention_mask == 0) | (attention_mask == 1))


def test_fluorescence_dataset():
    dataset = datasets.FluorescenceDataset(DATA_PATH, 'train')

    # Test 10 random accesses
    for _ in range(10):
        ids, attention_mask, fluorescence = dataset[random.randint(0, len(dataset) - 1)]
        assert isinstance(ids, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert isinstance(fluorescence, float)

        assert 0 <= ids.min() <= ids.max() < dataset.tokenizer.vocab_size
        assert np.all((attention_mask == 0) | (attention_mask == 1))


def test_stability_dataset():
    dataset = datasets.StabilityDataset(DATA_PATH, 'train')

    # Test 10 random accesses
    for _ in range(10):
        ids, attention_mask, stability = dataset[random.randint(0, len(dataset) - 1)]
        assert isinstance(ids, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert isinstance(stability, float)

        assert 0 <= ids.min() <= ids.max() < dataset.tokenizer.vocab_size
        assert np.all((attention_mask == 0) | (attention_mask == 1))


def test_proteinnet_dataset():
    dataset = datasets.ProteinnetDataset(DATA_PATH, 'train')

    # Test 10 random accesses
    for _ in range(10):
        ids, attention_mask, contact_map, valid_mask = dataset[random.randint(0, len(dataset) - 1)]
        assert isinstance(ids, np.ndarray)
        assert isinstance(attention_mask, np.ndarray)
        assert isinstance(contact_map, np.ndarray)
        assert isinstance(valid_mask, np.ndarray)

        assert 0 <= ids.min() <= ids.max() < dataset.tokenizer.vocab_size
        assert np.all((attention_mask == 0) | (attention_mask == 1))
        assert contact_map.shape == (valid_mask.size, valid_mask.size)
        assert np.all((valid_mask == 0) | (valid_mask == 1))
