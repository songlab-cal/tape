import typing
import argparse
import os
import multiprocessing as mp
import sys
import signal

import torch
import torch.distributed as dist
from torch.multiprocessing import _prctl_pr_set_pdeathsig  # type: ignore

from ..errors import EarlyStopping


def reduce_scalar(scalar: float) -> float:
    if dist.is_available() and dist.is_initialized():
        float_tensor = torch.cuda.FloatTensor([scalar])  # type: ignore
        dist.all_reduce(float_tensor)
        float_tensor /= dist.get_world_size()
        scalar = float_tensor.item()
    return scalar


def barrier_if_distributed() -> None:
    """Raises a barrier if in a distributed context, otherwise does nothing."""
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
    except EarlyStopping:
        sys.exit(signal.SIGUSR1)  # tape early stop exception
    except Exception:
        # Propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put(traceback.format_exc())
        sys.exit(1)


class ProcessContext:
    def __init__(self, processes, error_queues):
        self.error_queues = error_queues
        self.processes = processes
        self.sentinels = {
            process.sentinel: index
            for index, process in enumerate(processes)
        }

    def pids(self):
        return [int(process.pid) for process in self.processes]

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this process context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Arguments:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True

        # Wait for any process to fail or all of them to succeed.
        ready = mp.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
        # Assume failure. Terminate processes that are still alive.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()

        # There won't be an error on the queue if the process crashed.
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode == signal.SIGUSR1:
                return True
            elif exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise Exception(
                    "process %d terminated with signal %s" %
                    (error_index, name)
                )
            else:
                raise Exception(
                    "process %d terminated with exit code %d" %
                    (error_index, exitcode)
                )

        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
        raise Exception(msg)


def launch_process_group(func: typing.Callable,
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

    error_queues = []
    processes = []

    for local_rank in range(num_processes):
        # each process's rank
        dist_rank = num_processes * node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        args.local_rank = local_rank

        error_queue: mp.SimpleQueue[Exception] = mp.SimpleQueue()
        kwargs = {'args': args, 'env': current_env}
        process = mp.Process(
            target=_wrap,
            args=(func, kwargs, error_queue),
            daemon=daemon)
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    process_context = ProcessContext(processes, error_queues)
    if not join:
        return process_context

    while not process_context.join():
        pass
