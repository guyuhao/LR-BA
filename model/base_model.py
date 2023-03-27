import copy
import logging
import os

import torch
from torch import nn, optim
from torch.nn import init

from common.constants import CHECKPOINT_PATH


"""
Base class of all model architecture
"""


class BaseModel(nn.Module):
    def __init__(self, dataset, type, role, param_dict):
        nn.Module.__init__(self)
        self.dataset = dataset  # dataset name
        self.type = type  # model type, support bert, fcn, and resnet
        self.role = role  # model role, support top, passive, and active
        # self.pos = param_dict['pos']
        self.cuda = param_dict['cuda']  # whether to use cuda, 1 means use and 0 otherwise
        self.learning_rate = param_dict['lr']  # learning rate of the model

        self.optimizer = None
        self.scheduler = None


    def init_optim(self, param_dict, init_weight=True):
        """
        initialize optimizer

        :param dict param_dict: parameters of optimizer
        :param bool init_weight: whether to initialize model weights
        """
        parameters = self.parameters()
        if self.type == 'bert':
            parameters = [{"params": self.bert.parameters(), "lr": 5e-6},
                          {"params": self.linear.parameters(), "lr": 5e-4}]

        # default use SGD except that param_dict requires using Adam
        if 'optim' in param_dict and param_dict['optim'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=param_dict['lr'])
        else:
            self.optimizer = optim.SGD(parameters,
                                       momentum=param_dict['momentum'],
                                       weight_decay=param_dict['wd'],
                                       lr=param_dict['lr'])

        # use MultiStepLR scheduler only param_dict contains stone
        if 'stone' in param_dict:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=param_dict['stone'],
                                                                  gamma=param_dict['gamma'])

        if init_weight:
            # initialize model weights except for bert
            if self.type != 'bert':
                for m in self.modules():
                    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                        init.kaiming_normal_(m.weight)


    def save(self, name=None, id=None):
        """
        save model to local file located in CHECKPOINT_PATH/"dataset"

        :param str name: name of local file, use default name if not provided
        :param int id: id of passive party
        """
        path = '{}/{}'.format(CHECKPOINT_PATH, self.dataset)
        if name is None:
            if id is None:
                filepath = '{}/{}_{}'.format(path, self.role, self.type)  # for active party
            else:
                filepath = '{}/{}_{}_{}'.format(path, self.role, self.type, id)  # for passive party
        else:
            filepath = '{}/{}'.format(path, name)
        torch.save(self.state_dict(), filepath)

    def load(self, name=None, id=None):
        """
        load model from local file located in CHECKPOINT_PATH/"dataset"

        :param name: name of local file, use default name if not provided
        :param id: id of passive party
        :return: bool, whether to load successfully or not
        """
        path = '{}/{}'.format(CHECKPOINT_PATH, self.dataset)
        if name is None:
            if id is None:
                filepath = '{}/{}_{}'.format(path, self.role, self.type)  # for active party
            else:
                filepath = '{}/{}_{}_{}'.format(path, self.role, self.type, id)  # for passive party
        else:
            filepath = '{}/{}'.format(path, name)
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            self.load_state_dict(checkpoint)
            return True
        else:
            return False

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        if self.cuda:
            x = x.cuda()
        return self.classifier(x)

    def predict(self, x):
        return self.forward(x)

    def backward_(self):
        """
        backward using gradients of optimizer
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def backward(self, x, y, grads, epoch=0):
        """
        backward using given grads on y or on x if y is None

        :param x: backward on x, only works if y is None
        :param y: backward on y
        :param grads: gradients for backward
        :param epoch: current epoch
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if y is not None:
            y.backward(gradient=grads)
        else:
            x = x.float()
            if self.cuda:
                x = x.cuda()
            output = self.forward(x)
            output.backward(gradient=grads)

        self.optimizer.step()
        self.optimizer.zero_grad()

