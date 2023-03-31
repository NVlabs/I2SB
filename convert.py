# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from corruption import build_corruption
from dataset import imagenet
from i2sb import Runner, download_ckpt

import colored_traceback.always
from ipdb import set_trace as debug

import pickle
from easydict import EasyDict as edict

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        type=Path,   default=None,        help="resumed checkpoint name")

arg = parser.parse_args()

opt = edict()
opt.update(vars(arg))

ckpt_pkl = "results" / opt.ckpt / "options.pkl"
ckpt_pt  = "results" / opt.ckpt / "latest.pt"

with open(ckpt_pkl, "rb") as f:
    ckpt_opt = pickle.load(f)
print(ckpt_opt)

del ckpt_opt.n_gpu_per_node
del ckpt_opt.global_size
del ckpt_opt.ngc_job_id


with open(ckpt_pkl, "wb") as f:
    pickle.dump(ckpt_opt, f)

out = torch.load(ckpt_pt)
print(out.keys())
torch.save({"net": out["net"], "ema": out["ema"]}, ckpt_pt)

# out = torch.load("stage1_bwd.npz")
# out2 = {'net': out['z_b']['net'], 'ema': out['ema_b']}