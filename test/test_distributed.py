import torch
import torch.distributed as dist
import argparse
import multiprocessing as mp
import tape_pytorch.utils as utils
import os


def run_distributed(args: argparse.Namespace, env):
    os.environ = env
    device, n_gpu, is_master = utils.setup_distributed(args.local_rank, no_cuda=False)
    a = torch.LongTensor([args.local_rank]).cuda()
    dist.all_reduce(a, dist.ReduceOp.SUM)
    print(a)


def test_distributed():
    args = argparse.Namespace()
    args.test = 'hello'
    utils.spawn(run_distributed, args, 2)


if __name__ == '__main__':
    test_distributed()
