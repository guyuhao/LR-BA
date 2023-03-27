# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import init

from model.base_model import BaseModel

"""
Model architecture of the inference head, FCN-1
"""


class LrBaTopModel(BaseModel):
    def __init__(self, param_dict, ema=False):
        super(LrBaTopModel, self).__init__(
            dataset=param_dict['dataset'],
            type='fcn',
            role='top',
            param_dict=param_dict
        )
        self.classifier = nn.Sequential(
            # nn.ReLU(),
            nn.BatchNorm1d(param_dict['input_dim']),
            nn.ReLU(),
            nn.Linear(in_features=param_dict['input_dim'], out_features=param_dict['output_dim'])
        )
        if param_dict['cuda']:
            self.classifier = self.classifier.cuda()
        self.is_debug = False
        self.input_dim = param_dict['input_dim']

        for layer in self.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                init.ones_(layer.weight)

        if ema:
            for param in self.parameters():
                param.detach_()


def lr_ba_top_model(**kwargs):
    model = LrBaTopModel(**kwargs)
    return model
