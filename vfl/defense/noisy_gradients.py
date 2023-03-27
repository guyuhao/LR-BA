# -*- coding: utf-8 -*-
import numpy as np
import torch

"""
generate laplace noise according to scale, used for noisy gradient defense
"""


def noisy_count(noise_scale):
    beta = noise_scale
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    n_value = torch.tensor(n_value)
    return n_value


def laplace_noise(tensor, noise_scale):
    # generate noisy mask
    # whether the tensor to process is on cuda devices
    noisy_mask = torch.zeros(tensor.shape).to(torch.float)
    if 'cuda' in str(tensor.device):
        noisy_mask = noisy_mask.cuda()
    noisy_mask = noisy_mask.flatten()
    for i in range(noisy_mask.shape[0]):
        noisy_mask[i] = noisy_count(noise_scale)
    noisy_mask = noisy_mask.reshape(tensor.shape)
    tensor = tensor + noisy_mask
    return tensor
