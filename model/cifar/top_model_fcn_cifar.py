# -*- coding: utf-8 -*-

from torch import nn

from model.base_model import BaseModel

"""
Top model architecture for CIFAR, FCN-4
"""


class TopModelFcnCifar(BaseModel):
    def __init__(self, param_dict):
        super(TopModelFcnCifar, self).__init__(
            dataset=param_dict['dataset'],
            type='fcn',
            role='top',
            param_dict=param_dict
        )
        if param_dict['dataset'] == 'cifar10':
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(param_dict['input_dim']),
                nn.ReLU(),
                nn.Linear(in_features=param_dict['input_dim'], out_features=20),
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(in_features=20, out_features=10),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Linear(in_features=10, out_features=10),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Linear(in_features=10, out_features=param_dict['output_dim'])
            )
        else:  # CIFAR100
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(param_dict['input_dim']),
                nn.ReLU(),
                nn.Linear(in_features=param_dict['input_dim'], out_features=200),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(in_features=200, out_features=100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(in_features=100, out_features=100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(in_features=100, out_features=param_dict['output_dim'])
            )
        if param_dict['cuda']:
            self.classifier = self.classifier.cuda()
        self.is_debug = False

        self.init_optim(param_dict)


def top_model_fcn_cifar(**kwargs):
    model = TopModelFcnCifar(**kwargs)
    return model
