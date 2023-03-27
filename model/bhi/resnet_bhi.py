# -*- coding: utf-8 -*-
import torch

from model.base_model import BaseModel
from model.cifar.resnet import ResNet, BasicBlock

"""
Bottom model architecture for BHI, Resnet20
"""


class ResnetBhi(BaseModel, ResNet):
    def __init__(self, param_dict):
        BaseModel.__init__(
            self,
            dataset=param_dict['dataset'],
            type='resnet',
            role=param_dict['role'],
            param_dict=param_dict
        )
        ResNet.__init__(
            self,
            block=BasicBlock,
            num_blocks=[3, 3, 3],
            kernel_size=(3, 3),
            num_classes=param_dict['output_dim']
        )
        if self.cuda:
            self.to('cuda')
        self.is_debug = False

        self.output_dim = param_dict['output_dim']

        self.init_optim(param_dict)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        if self.cuda:
            x = x.cuda()
        return ResNet.forward(self, x=x)

    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        if self.cuda:
            x = x.cuda()
        return ResNet.forward(self, x=x)


def resnet_bhi(**kwargs):
    model = ResnetBhi(**kwargs)
    return model
