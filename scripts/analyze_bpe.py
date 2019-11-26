from pathlib import Path
from collections import Counter
import numpy as np
from tape_pytorch.tokenizers import PfamTokenizer
from tape_pytorch.datasets import TAPEDataset
from tqdm import tqdm
import itertools

DATA_PATH = Path(__file__).parent.parent / 'data'

tokenizer = PfamTokenizer(model_file=DATA_PATH / 'pfam.model')
dataset = TAPEDataset(
    DATA_PATH,
    'secondary_structure/secondary_structure_train.lmdb',
    tokenizer=tokenizer)

with (DATA_PATH / 'pfam.vocab').open() as f:
    tokens = [line.split()[0] for line in f]

while tokens[0][0] == "<":  # remove special tokens
    tokens.pop(0)

token_lengths = Counter([len(token) for token in tokens])

length_probs = {length: freq / sum(token_lengths.values())
                for length, freq in token_lengths.items()}
lengths = list(length_probs.keys())
probs = list(length_probs.values())


def select_random_token_length():
    return np.random.choice(lengths, p=probs)


def generate_random_length_sequence(seqlen: int):
    length_sequence = []
    total = 0
    while total < seqlen:
        length = select_random_token_length()
        total += length
        length_sequence.append(length)

    total -= length
    length_sequence.pop(-1)

    if total < seqlen:
        length_sequence.append(seqlen - total)

    return length_sequence


def get_piece_lengths(indices):
    indices = indices[1:-1]  # remove cls/sep tokens
    pieces = tokenizer.convert_ids_to_tokens(indices.tolist())
    piece_lengths = [len(piece) for piece in pieces]
    piece_lengths[0] -= 1  # sentencepiece inserts a start token
    if piece_lengths[0] == 0:
        piece_lengths = piece_lengths[1:]

    return piece_lengths


def find_overlap(sequence, piece_lengths):
    end_indices = np.cumsum(piece_lengths)
    start_indices = end_indices - end_indices[0]

    overlap = [(sequence[start], np.any(sequence[start:end] != sequence[start]))
               for start, end in zip(start_indices, end_indices)]
    return overlap


true_3_overlap = list(itertools.chain.from_iterable(
    find_overlap(item['ss3'], get_piece_lengths(indices))
    for item, indices, _ in tqdm(dataset)))

rand_3_overlap = list(itertools.chain.from_iterable(
    find_overlap(item['ss3'], generate_random_length_sequence(item['protein_length']))
    for item, indices, _ in tqdm(dataset)))

# true_8_overlap = list(itertools.chain.from_iterable(
    # find_overlap(item['ss8'], get_piece_lengths(indices))
    # for item, indices, _ in tqdm(dataset)))

# rand_8_overlap = list(itertools.chain.from_iterable(
    # find_overlap(item['ss8'], generate_random_length_sequence(item['protein_length']))
    # for item, indices, _ in tqdm(dataset)))
