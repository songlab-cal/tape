from typing import Dict, Any
import numpy as np
import csv
from tape.datasets import LMDBDataset
from pathlib import Path
import argparse
from tqdm import tqdm
import json

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path, help="Path to existing lmdb file")
parser.add_argument("output", type=Path, help="Path to output csv file")
args = parser.parse_args()


data = LMDBDataset(args.input)
fieldnames = list(data[0].keys())
fieldnames.remove("id")
fieldnames.insert(0, "id")
fieldnames.remove("primary")
fieldnames.insert(1, "primary")
fieldnames.remove("protein_length")
fieldnames.insert(2, "protein_length")


def convert_type(key: str, val: Any) -> str:
    if key == "ss3":
        SS_3_DICT = {0: "H", 1: "E", 2: "C"}
        val = "".join(map(SS_3_DICT.__getitem__, val))
    elif key == "ss8":
        SS_8_DICT = {
            0: "G",
            1: "H",
            2: "I",
            3: "B",
            4: "E",
            5: "S",
            6: "T",
            7: "C",
        }
        val = "".join(map(SS_8_DICT.__getitem__, val))
    elif isinstance(val, np.ndarray):
        val = val.tolist()
    elif isinstance(val, bytes):
        val = val.decode()
    return json.dumps(val)


def convert_numpy(d: Dict[str, Any]) -> Dict[str, str]:
    return {key: convert_type(key, val) for key, val in d.items()}


with args.output.open("w") as f:
    writer = csv.DictWriter(f, fieldnames)
    writer.writeheader()
    writer.writerows(map(convert_numpy, tqdm(data)))
