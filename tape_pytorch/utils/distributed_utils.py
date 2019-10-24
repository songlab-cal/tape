import torch
import torch.distributed as dist


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
