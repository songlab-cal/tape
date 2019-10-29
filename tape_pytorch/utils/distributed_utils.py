import typing
import argparse
import os
import multiprocessing as mp
import sys
import signal

import torch
import torch.distributed as dist
from torch.multiprocessing import _prctl_pr_set_pdeathsig
from torch.multiprocessing.spawn import SpawnContext


def reduce_scalar(scalar: float) -> float:
    if dist.is_available() and dist.is_initialized():
        float_tensor = torch.cuda.FloatTensor([scalar])  # type: ignore
        dist.all_reduce(float_tensor)
        float_tensor /= dist.get_world_size()
        scalar = float_tensor.item()
    return scalar


def barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _wrap(fn, kwargs, error_queue):
    # prctl(2) is a Linux specific system call.
    # On other systems the following function call has no effect.
    # This is set to ensure that non-daemonic child processes can
    # terminate if their parent terminates before they do.
    _prctl_pr_set_pdeathsig(signal.SIGINT)

    try:
        fn(**kwargs)
    except KeyboardInterrupt:
        pass  # SIGINT; Killed by parent, do nothing
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


def spawn(func: typing.Callable,
          args: argparse.Namespace,
          num_processes: int,
          num_nodes: int = 1,
          node_rank: int = 0,
          master_addr: str = "127.0.0.1",
          master_port: int = 29500,
          join: bool = True,
          daemon: bool = False):
    # world size in terms of number of processes
    dist_world_size = num_processes * num_nodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = master_addr
    current_env["MASTER_PORT"] = str(master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    if 'OMP_NUM_THREADS' not in os.environ and num_processes > 1:
        current_env["OMP_NUM_THREADS"] = str(4)

    ctx = mp.get_context('spawn')
    error_queues = []
    processes = []

    for local_rank in range(num_processes):
        # each process's rank
        dist_rank = num_processes * node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        args.local_rank = local_rank

        error_queue = ctx.SimpleQueue()
        kwargs = {'args': args, 'env': current_env}
        process = ctx.Process(
            target=_wrap,
            args=(func, kwargs, error_queue),
            daemon=daemon)
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    spawn_context = SpawnContext(processes, error_queues)
    if not join:
        return spawn_context

    while not spawn_context.join():
        pass
