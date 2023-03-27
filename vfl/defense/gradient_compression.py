#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gyh
@File    ：gradient_compression.py
@Author  ：Gu Yuhao
@Date    ：2022/4/18 下午3:41
Implementation of gradient compression, refers to "Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training"
"""
import torch


class GradientCompression:
    def __init__(self, gc_percent):
        self.thresh_hold = 0.
        self.gc_percent = gc_percent

    def update_thresh_hold(self, tensor):
        tensor_copy = tensor.clone().detach()
        tensor_copy = torch.abs(tensor_copy)
        survivial_values = torch.topk(tensor_copy.reshape(1, -1),
                                      int(tensor_copy.reshape(1, -1).shape[1] * (1-self.gc_percent)))
        self.thresh_hold = survivial_values[0][0][-1]

    def prune_tensor(self, tensor):
        background_tensor = torch.zeros(tensor.shape).to(torch.float)
        if 'cuda' in str(tensor.device):
            background_tensor = background_tensor.cuda()
        tensor = torch.where(abs(tensor) > self.thresh_hold, tensor, background_tensor)
        return tensor

    def compression(self, gradients):
        self.update_thresh_hold(gradients)
        result = self.prune_tensor(gradients)
        return result
