# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os

import numpy as np
import torch

import torch.distributed as dist
from torch.multiprocessing import Process

def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    dist.barrier()
    cleanup()

def cleanup():
    dist.destroy_process_group()

def average_grads(params):
    size = float(dist.get_world_size())
    for param in params:
        if param.requires_grad:
            # _average_tensor(param.grad, size)
            with torch.no_grad():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size

def average_params(params):
    size = float(dist.get_world_size())
    for param in params:
        # _average_tensor(param, size)
        with torch.no_grad():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= size

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)

def all_gather(tensor, log=None):
    if log: log.info("Gathering tensor across {} devices... ".format(dist.get_world_size()))
    gathered_tensors = [
        torch.zeros_like(tensor) for _ in range(dist.get_world_size())
    ]
    with torch.no_grad():
        dist.all_gather(gathered_tensors, tensor)
    return gathered_tensors
