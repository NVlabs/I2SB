# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

import torch
import torchvision
from cleanfid.resize import build_resizer
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_batch_features, frechet_distance

FID_REF_DIR = Path("data")

def collect_features(dataset, mode, batch_size,
                     num_workers, device=torch.device("cuda"), use_dataparallel=True, verbose=True):

    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )
    feat_model = build_feature_extractor(mode, device, use_dataparallel=use_dataparallel)
    l_feats = []
    pbar = tqdm(dataloader, desc="FID") if verbose else dataloader
    for batch in pbar:
        l_feats.append(get_batch_features(batch, feat_model, device))

    np_feats = np.concatenate(l_feats)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    return mu, sigma

class NumpyResizeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mode, size=(299, 299)):
        self.dataset = dataset
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def get_img_np(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img_np = self.get_img_np(i)

        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        # ToTensor() converts to [0,1] only if input in uint8
        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized))*255
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t

@torch.no_grad()
def compute_fid_from_numpy(numpy_arr, ref_mu, ref_sigma, batch_size=256, mode="legacy_pytorch"):

    dataset = NumpyResizeDataset(numpy_arr, mode=mode)
    mu, sigma = collect_features(dataset, mode,
        num_workers=1, batch_size=batch_size, use_dataparallel=False, verbose=False,
    )
    return frechet_distance(mu, sigma, ref_mu, ref_sigma)

class LMDBResizeDataset(NumpyResizeDataset):
    def __init__(self, dataset, mode):
        super(LMDBResizeDataset, self).__init__(dataset, mode)

    def get_img_np(self, i):
        img_pil, _ = self.dataset[i]
        return np.array(img_pil)

def compute_fid_ref_stat(opt, log):
    from dataset import imagenet
    from torchvision import transforms

    mode = opt.mode

    # build dataset
    transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
    ])
    lmdb_dataset = imagenet.build_lmdb_dataset(opt, log, train=opt.split=="train", transform=transform)
    dataset = LMDBResizeDataset(lmdb_dataset, mode=mode)
    log.info(f"[FID] Built Imagenet {opt.split} dataset, size={len(dataset)}!")

    # compute fid statistics
    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)
    mu, sigma = collect_features(dataset, mode, batch_size=512, num_workers=num_workers)
    log.info(f"Collected inception features, {mu.shape=}, {sigma.shape=}!")

    # save and return statistics
    os.makedirs(FID_REF_DIR, exist_ok=True)
    fn = FID_REF_DIR / f"fid_imagenet_{opt.image_size}_{opt.split}.npz"
    np.savez(fn, mu=mu, sigma=sigma)
    log.info(f"Saved FID reference statistics to {fn}!")
    return mu, sigma


if __name__ == '__main__':
    import argparse
    from logger import Logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--split",       type=str,  choices=["train", "val"], help="which dataset to compute FID ref statistics")
    parser.add_argument("--mode",        type=str,  default="legacy_pytorch", help="the FID computation mode used in clean-fid")
    parser.add_argument("--dataset-dir", type=Path, default="/dataset",       help="path to LMDB dataset")
    parser.add_argument("--image-size",  type=int,  default=256)
    opt = parser.parse_args()

    log = Logger(0, ".log")
    log.info(f"======== Compute FID ref statistics: mode={opt.mode} ========")
    compute_fid_ref_stat(opt, log)
