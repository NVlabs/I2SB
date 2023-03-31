# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from diffusion_palette_eval.
#
# Source:
# https://bit.ly/eval-pix2pix
#
# ---------------------------------------------------------------

import io
import math
from PIL import Image, ImageDraw

import os

import numpy as np
import torch

from pathlib import Path
import gdown
from ipdb import set_trace as debug

FREEFORM_URL = "https://drive.google.com/file/d/1-5YRGsekjiRKQWqo0BV5RVQu0bagc12w/view?usp=share_link"

# code adoptted from
# https://bit.ly/eval- pix2pix
def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask

# code adoptted from
# https://bit.ly/eval-pix2pix
def load_masks(filename):
    # filename = "imagenet_freeform_masks.npz"
    shape = [10000, 256, 256]

    # shape = [10950, 256, 256] # Uncomment this for places2.

    # Load the npz file.
    with open(filename, 'rb') as f:
        data = f.read()

    data = dict(np.load(io.BytesIO(data)))
    # print("Categories of masks:")
    # for key in data:
    #     print(key)

    # Unpack and reshape the masks.
    for key in data:
        data[key] = np.unpackbits(data[key], axis=None)[:np.prod(shape)].reshape(shape).astype(np.uint8)

    # data[key] contains [10000, 256, 256] array i.e. 10000 256x256 masks.
    return data

def load_freeform_masks(op_type):
    data_dir = Path("data")

    mask_fn = data_dir / f"imagenet_{op_type}_masks.npz"
    if not mask_fn.exists():
        # download orignal npz from palette google drive
        orig_mask_fn = str(data_dir / "imagenet_freeform_masks.npz")
        if not os.path.exists(orig_mask_fn):
            gdown.download(url=FREEFORM_URL, output=orig_mask_fn, quiet=False, fuzzy=True)
        masks = load_masks(orig_mask_fn)

        # store freeform of current ratio for faster loading in future
        key = {
            "freeform1020": "10-20% freeform",
            "freeform2030": "20-30% freeform",
            "freeform3040": "30-40% freeform",
        }.get(op_type)
        np.savez(mask_fn, mask=masks[key])

    # [10000, 256, 256] --> [10000, 1, 256, 256]
    return np.load(mask_fn)["mask"][:,None]

def get_center_mask(image_size):
    h, w = image_size
    mask = bbox2mask(image_size, (h//4, w//4, h//2, w//2))
    return torch.from_numpy(mask).permute(2,0,1)

def build_inpaint_center(opt, log, mask_type):
    assert mask_type == "center"

    log.info(f"[Corrupt] Inpaint: {mask_type=}  ...")

    center_mask = get_center_mask([opt.image_size, opt.image_size])[None,...] # [1,1,256,256]
    center_mask = center_mask.to(opt.device)

    def inpaint_center(img):
        # img: [-1,1]
        mask = center_mask
        # img[mask==0] = img[mask==0], img[mask==1] = 1 (white)
        return img * (1. - mask) + mask, mask

    return inpaint_center

def build_inpaint_freeform(opt, log, mask_type):
    assert "freeform" in mask_type

    log.info(f"[Corrupt] Inpaint: {mask_type=}  ...")

    freeform_masks = load_freeform_masks(mask_type) # [10000, 1, 256, 256]
    n_freeform_masks = freeform_masks.shape[0]
    freeform_masks = torch.from_numpy(freeform_masks).to(opt.device)

    def inpaint_freeform(img):
        # img: [-1,1]
        index = np.random.randint(n_freeform_masks, size=img.shape[0])
        mask = freeform_masks[index]
        # img[mask==0] = img[mask==0], img[mask==1] = 1 (white)
        return img * (1. - mask) + mask, mask

    return inpaint_freeform