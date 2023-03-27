# -*- coding: utf-8 -*-
import torch

from model.base_model import BaseModel
from model.yahoo.mix_text import MixText

"""
Bottom model architecture for Yahoo, Bert
"""


class BertYahoo(BaseModel, MixText):
    def __init__(self, param_dict):
        BaseModel.__init__(
            self,
            dataset=param_dict['dataset'],
            type='bert',
            role=param_dict['role'],
            param_dict=param_dict
        )
        MixText.__init__(
            self,
            num_labels=param_dict['num_classes'],
            mix_option=True
        )
        if self.cuda:
            self.to('cuda')
        self.is_debug = False

        self.output_dim = param_dict['output_dim']

        self.init_optim(param_dict)

    def forward(self, x, x2=None, l=None, mix_layer=1000):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if self.cuda:
            x = x.cuda()
        return MixText.forward(self, x=x, x2=x2, l=l, mix_layer=mix_layer)

    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if self.cuda:
            x = x.cuda()
        return MixText.forward(self, x=x)


def bert_yahoo(**kwargs):
    model = BertYahoo(**kwargs)
    return model
