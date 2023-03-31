# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from ddrm-jpeg.
#
# Source:
# https://github.com/bahjat-kawar/ddrm-jpeg/blob/master/functions/jpeg_torch.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_DDRM_JPEG).
# The modifications to this file are subject to the same license.
# ---------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn

def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1))[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)


def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)


def torch_rgb2ycbcr(x):
    # Assume x is a batch of size (N x C x H x W)
    v = torch.tensor([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]]).to(x.device)
    ycbcr = torch.tensordot(x, v, dims=([1], [1])).transpose(3, 2).transpose(2, 1)
    ycbcr[:,1:] += 128
    return ycbcr


def torch_ycbcr2rgb(x):
    # Assume x is a batch of size (N x C x H x W)
    v = torch.tensor([[ 1.00000000e+00, -3.68199903e-05,  1.40198758e+00],
       [ 1.00000000e+00, -3.44113281e-01, -7.14103821e-01],
       [ 1.00000000e+00,  1.77197812e+00, -1.34583413e-04]]).to(x.device)
    x[:, 1:] -= 128
    rgb = torch.tensordot(x, v, dims=([1], [1])).transpose(3, 2).transpose(2, 1)
    return rgb

def chroma_subsample(x):
    return x[:, 0:1, :, :], x[:, 1:, ::2, ::2]


def general_quant_matrix(qf = 10):
    q1 = torch.tensor([
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
    ])
    q2 = torch.tensor([
        17,  18,  24,  47,  99,  99,  99,  99,
        18,  21,  26,  66,  99,  99,  99,  99,
        24,  26,  56,  99,  99,  99,  99,  99,
        47,  66,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99,
        99,  99,  99,  99,  99,  99,  99,  99
    ])
    s = (5000 / qf) if qf < 50 else (200 - 2 * qf)
    q1 = torch.floor((s * q1 + 50) / 100)
    q1[q1 <= 0] = 1
    q1[q1 > 255] = 255
    q2 = torch.floor((s * q2 + 50) / 100)
    q2[q2 <= 0] = 1
    q2[q2 > 255] = 255
    return q1, q2


def quantization_matrix(qf):
    return general_quant_matrix(qf)
    # q1 = torch.tensor([[ 80,  55,  50,  80, 120, 200, 255, 255],
    #                    [ 60,  60,  70,  95, 130, 255, 255, 255],
    #                    [ 70,  65,  80, 120, 200, 255, 255, 255],
    #                    [ 70,  85, 110, 145, 255, 255, 255, 255],
    #                    [ 90, 110, 185, 255, 255, 255, 255, 255],
    #                    [120, 175, 255, 255, 255, 255, 255, 255],
    #                    [245, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255]])
    # q2 = torch.tensor([[ 85,  90, 120, 235, 255, 255, 255, 255],
    #                    [ 90, 105, 130, 255, 255, 255, 255, 255],
    #                    [120, 130, 255, 255, 255, 255, 255, 255],
    #                    [235, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255],
    #                    [255, 255, 255, 255, 255, 255, 255, 255]])
    # return q1, q2

def jpeg_encode(x, qf):
    # Assume x is a batch of size (N x C x H x W)
    # [-1, 1] to [0, 255]
    x = (x + 1) / 2 * 255
    n_batch, _, n_size, _ = x.shape

    x = torch_rgb2ycbcr(x)
    x_luma, x_chroma = chroma_subsample(x)
    unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_chroma = unfold(x_chroma).transpose(2, 1)

    x_luma = x_luma.reshape(-1, 8, 8) - 128
    x_chroma = x_chroma.reshape(-1, 8, 8) - 128

    dct_layer = LinearDCT(8, 'dct', norm='ortho')
    dct_layer.to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)

    x_luma = x_luma.view(-1, 1, 8, 8)
    x_chroma = x_chroma.view(-1, 2, 8, 8)

    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma /= q1.view(1, 8, 8)
    x_chroma /= q2.view(1, 8, 8)

    x_luma = x_luma.round()
    x_chroma = x_chroma.round()

    x_luma = x_luma.reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = x_chroma.reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)

    fold = nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)

    return [x_luma, x_chroma]



def jpeg_decode(x, qf):
    # Assume x[0] is a batch of size (N x 1 x H x W) (luma)
    # Assume x[1:] is a batch of size (N x 2 x H/2 x W/2) (chroma)
    x_luma, x_chroma = x
    n_batch, _, n_size, _ = x_luma.shape
    unfold = nn.Unfold(kernel_size=(8, 8), stride=(8, 8))
    x_luma = unfold(x_luma).transpose(2, 1)
    x_luma = x_luma.reshape(-1, 1, 8, 8)
    x_chroma = unfold(x_chroma).transpose(2, 1)
    x_chroma = x_chroma.reshape(-1, 2, 8, 8)

    q1, q2 = quantization_matrix(qf)
    q1 = q1.to(x_luma.device)
    q2 = q2.to(x_luma.device)
    x_luma *= q1.view(1, 8, 8)
    x_chroma *= q2.view(1, 8, 8)

    x_luma = x_luma.reshape(-1, 8, 8)
    x_chroma = x_chroma.reshape(-1, 8, 8)

    dct_layer = LinearDCT(8, 'idct', norm='ortho')
    dct_layer.to(x_luma.device)
    x_luma = apply_linear_2d(x_luma, dct_layer)
    x_chroma = apply_linear_2d(x_chroma, dct_layer)

    x_luma = (x_luma + 128).reshape(n_batch, (n_size // 8) ** 2, 64).transpose(2, 1)
    x_chroma = (x_chroma + 128).reshape(n_batch, (n_size // 16) ** 2, 64 * 2).transpose(2, 1)

    fold = nn.Fold(output_size=(n_size, n_size), kernel_size=(8, 8), stride=(8, 8))
    x_luma = fold(x_luma)
    fold = nn.Fold(output_size=(n_size // 2, n_size // 2), kernel_size=(8, 8), stride=(8, 8))
    x_chroma = fold(x_chroma)

    x_chroma_repeated = torch.zeros(n_batch, 2, n_size, n_size, device = x_luma.device)
    x_chroma_repeated[:, :, 0::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 0::2, 1::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 0::2] = x_chroma
    x_chroma_repeated[:, :, 1::2, 1::2] = x_chroma

    x = torch.cat([x_luma, x_chroma_repeated], dim=1)

    x = torch_ycbcr2rgb(x)

    # [0, 255] to [-1, 1]
    x = x / 255 * 2 - 1

    return x


def build_jpeg(log, qf):
    log.info(f"[Corrupt] JPEG restoration: {qf=}  ...")
    def jpeg(img):
        return jpeg_decode(jpeg_encode(img, qf), qf)
    return jpeg
