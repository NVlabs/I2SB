# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch

import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import resnet50

from ipdb import set_trace as debug

class ImageNormalizer(torch.nn.Module):

    def __init__(self, mean, std) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, image):
        # note: image should be in [-1,1]
        image = (image+1)/2 # [-1,1] -> [0,1]
        image = F.interpolate(image, size=(224, 224), mode='bicubic')
        return (image - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore

def normalize_model(model, mean, std):
    layers = OrderedDict([('normalize', ImageNormalizer(mean, std)),
                          ('model', model)])
    return torch.nn.Sequential(layers)

def build_resnet50():
    model = resnet50(pretrained=True)
    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)
    model = normalize_model(model, mu, sigma)
    model.eval()
    return model
