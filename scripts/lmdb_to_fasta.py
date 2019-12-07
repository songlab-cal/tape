import argparse
import math
from tqdm import tqdm
from Bio.SeqIO.FastaIO import Seq, SeqRecord
from tape.datasets import LMDBDataset

parser = argparse.ArgumentParser(description='Convert an lmdb file into a fasta file')
parser.add_argument('lmdbfile', type=str, help='The lmdb file to convert')
parser.add_argument('fastafile', type=str, help='The fasta file to output')
args = parser.parse_args()

dataset = LMDBDataset(args.lmdbfile)

id_fill = math.ceil(math.log10(len(dataset)))

fastafile = args.fastafile
if not fastafile.endswith('.fasta'):
    fastafile += '.fasta'

with open(fastafile, 'w') as outfile:
    for i, element in enumerate(tqdm(dataset)):
        id_ = element.get('id', str(i).zfill(id_fill))
        if isinstance(id_, bytes):
            id_ = id_.decode()

        primary = element['primary']
        seq = Seq(primary)
        record = SeqRecord(seq, id_)
        outfile.write(record.format('fasta'))
