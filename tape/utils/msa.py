from typing import Tuple, List, Dict, Optional
import tempfile
import string
from pathlib import Path
from Bio import SeqIO
import subprocess
from .typing_utils import PathLike


def parse_fasta(
    filename: PathLike,
    remove_insertions: bool = False,
    remove_gaps: bool = False,
) -> Tuple[List[str], List[str]]:

    filename = Path(filename)
    if filename.suffix == ".sto":
        form = "stockholm"
    elif filename.suffix in (".fas", ".fasta", ".a3m"):
        form = "fasta"
    else:
        raise ValueError(f"Unknown file format {filename.suffix}")

    translate_dict: Dict[str, Optional[str]] = {}
    if remove_insertions:
        translate_dict.update(dict.fromkeys(string.ascii_lowercase))
    else:
        translate_dict.update(dict(zip(string.ascii_lowercase, string.ascii_uppercase)))

    if remove_gaps:
        translate_dict["-"] = None

    translate_dict["."] = None
    translate_dict["*"] = None
    translation = str.maketrans(translate_dict)

    def process_record(record: SeqIO.SeqRecord):
        return record.description, str(record.seq).translate(translation)

    records = SeqIO.parse(str(filename), form)
    records = map(process_record, records)
    records = zip(*records)
    headers, sequences = tuple(records)
    return headers, sequences


def hhfilter(
    filename: PathLike,
    hhfilter_bin: str = "hhfilter",
    id: int = 90,
    diff: int = 100,
    cov: int = 0,
    qid: int = 0,
    qsc: float = -20.0,
) -> List[Tuple[int, str]]:
    command = (
        "bash -c '"
        f"{hhfilter_bin} -i {filename} -M a3m -o >(cat) "
        f"-id {id} "
        f"-diff {diff} "
        f"-cov {cov} "
        f"-qid {qid} "
        f"-qsc {qsc}"
        "'"
    )
    p = subprocess.run(command, shell=True, capture_output=True)
    p.check_returncode()
    stdout = p.stdout.decode().strip().split("\n")
    return [
        (int(header[1:]), seq) for header, seq in zip(stdout[::2], stdout[1::2])
    ]


def hhfilter_sequences(
    sequences: List[str],
    hhfilter_bin: str = "hhfilter",
    id: int = 90,
    diff: int = 100,
    cov: int = 0,
    qid: int = 0,
    qsc: float = -20.0,
) -> List[Tuple[int, str]]:
    with tempfile.NamedTemporaryFile("w") as fh:
        fh.write("\n".join(f">{i}\n{seq}" for i, seq in enumerate(sequences)))
        fh.flush()
        return hhfilter(fh.name, hhfilter_bin, id, diff, cov, qid, qsc)
