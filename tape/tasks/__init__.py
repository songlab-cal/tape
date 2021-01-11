from argparse import ArgumentParser
from .tape_task import TAPETask


def get(parser: ArgumentParser) -> TAPETask:
    known_args, _ = parser.parse_known_args()
    task_name = parser.parse_known_args()[0].task
    if task_name == "secondary_structure":
        from .secondary_structure_task import SecondaryStructureTask
        return SecondaryStructureTask
    elif task_name == "fluorescence":
        from .fluorescence_task import FluorescenceTask
        return FluorescenceTask
    else:
        raise ValueError(f"Unrecognized task {task_name}")
