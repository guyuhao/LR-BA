# -*- coding: utf-8 -*-

from torch import nn

from model.base_model import BaseModel

"""
Top model architecture for NUS-WDIE, FCN-1
"""


class TopModelFcnNusWide(BaseModel):
    def __init__(self, param_dict):
        super(TopModelFcnNusWide, self).__init__(
            dataset=param_dict['dataset'],
            type='fcn',
            role='top',
            param_dict=param_dict
        )
        self.classifier = nn.Sequential(
            # nn.ReLU(),
            # nn.Linear(in_features=param_dict['input_dim'], out_features=16),
            # nn.ReLU(),
            # nn.Linear(in_features=16, out_features=8),
            # nn.ReLU(),
            # nn.Linear(in_features=8, out_features=param_dict['output_dim'])
            nn.BatchNorm1d(param_dict['input_dim']),
            nn.ReLU(),
            nn.Linear(in_features=param_dict['input_dim'], out_features=param_dict['output_dim'])
        )
        if param_dict['cuda']:
            self.classifier = self.classifier.cuda()
        self.is_debug = False

        self.init_optim(param_dict, init_weight=False)


def top_model_fcn_nus_wide(**kwargs):
    model = TopModelFcnNusWide(**kwargs)
    return model
