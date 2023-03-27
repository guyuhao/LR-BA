# -*- coding: utf-8 -*-
import copy

from torch import nn, optim

from model.base_model import BaseModel


"""
Bottom model architecture for NUS-WIDE, FCN-1
"""


class FcnNusWide(BaseModel):
    def __init__(self, param_dict):
        super(FcnNusWide, self).__init__(
            dataset='nus_wide',
            type='fcn',
            role=param_dict['role'],
            param_dict=param_dict
        )
        self.classifier = nn.Sequential(
            # nn.Linear(in_features=param_dict['input_dim'], out_features=64),
            # nn.ReLU(),
            # nn.Linear(in_features=64, out_features=16),
            # nn.ReLU(),
            # nn.Linear(in_features=16, out_features=param_dict['output_dim'])
            nn.Linear(in_features=param_dict['input_dim'], out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=param_dict['output_dim'])
        )
        if self.cuda:
            self.classifier = self.classifier.cuda()
        self.is_debug = False

        self.output_dim = param_dict['output_dim']

        self.init_optim(param_dict, init_weight=False)

def fcn_nus_wide(**kwargs):
    model = FcnNusWide(**kwargs)
    return model
