from tape.datasets import LMDBDataset
from mogwai.alignment import make_a3m
from pathlib import Path
import tempfile
import submitit
import argparse
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument(
    "tape_data",
    type=str,
    help="Path to TAPE secondary structure datasets.",
)
parser.add_argument("uniclust_database", type=str, help="Path to uniclust database")
parser.add_argument(
    "--debug",
    action="store_true",
    help="Just runs one local conversion before stopping.",
)

args = parser.parse_args()
data_dir = Path(args.tape_data)
a3m_dir = data_dir / "a3m"
a3m_dir.mkdir(exist_ok=True)


def create_a3m(id_: str, sequence: str) -> None:
    output_file = a3m_dir / f"{id_}.a3m"
    if output_file.exists():
        return
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        fasta = path / f"{id_}.fasta"
        fasta.write_text(f">{id_}\n{sequence}")
        make_a3m(fasta, args.uniclust_database)
        a3m = fasta.with_suffix(".a3m")
        a3m.rename(output_file)


if args.debug:
    data = LMDBDataset(data_dir / "secondary_structure_train.lmdb")
    create_a3m(data[0]["id"], data[0]["primary"])

# executor = submitit.AutoExecutor("netsurf-msa")
# executor.update_parameters(
    # timeout_min=60,
    # mem_gb=64,
    # cpus_per_task=20,
    # slurm_array_parallelism=128,
# )

# with executor.batch():
for data_file in data_dir.glob("*.lmdb"):
    data = LMDBDataset(data_file)
    for item in data:  # type: ignore
        id_ = item["id"].decode()
        seq = item["primary"]
        create_a3m(id_, seq)
        # executor.submit(create_a3m, id_, seq)
