# -*- coding: utf-8 -*-

from torch import nn

from model.base_model import BaseModel

"""
Top model architecture for BHI, FCN-4
"""


class TopModelFcnYahoo(BaseModel):
    def __init__(self, param_dict):
        super(TopModelFcnYahoo, self).__init__(
            dataset=param_dict['dataset'],
            type='fcn',
            role='top',
            param_dict=param_dict
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(param_dict['input_dim']),
            nn.ReLU(),
            nn.Linear(in_features=param_dict['input_dim'], out_features=10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=param_dict['output_dim'])
        )
        if param_dict['cuda']:
            self.classifier = self.classifier.cuda()
        self.is_debug = False

        self.init_optim(param_dict)


def top_model_fcn_yahoo(**kwargs):
    model = TopModelFcnYahoo(**kwargs)
    return model
