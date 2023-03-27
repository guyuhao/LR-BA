# -*- coding: utf-8 -*-
import torch

"""
clip tensor with threshold, used for clip defense
"""


def norm_clip(parameters, max_norm, norm_type=2):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = torch.norm(parameters.detach(), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        parameters.detach().mul_(clip_coef.to(parameters.device))
    return
