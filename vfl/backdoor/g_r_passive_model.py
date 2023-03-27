# -*- coding: utf-8 -*-
import logging
import math
import random

import torch

from vfl.party_models import VFLPassiveModel

"""
Malicious passive party for gradient-replacement backdoor
"""

class GRPassiveModel(VFLPassiveModel):

    def __init__(self, bottom_model, amplify_ratio=1, top_trainable=0):
        self.backdoor_indices = None
        self.target_grad = None
        self.target_indices = None
        self.amplify_ratio = amplify_ratio
        self.top_trainable = top_trainable
        self.components = None
        self.is_debug = False
        self.pair_set = dict()
        self.target_gradients = dict()
        self.backdoor_X = dict()
        super().__init__(bottom_model)

    def set_epoch(self, epoch):
        self.epoch = epoch
        # self.target_grad = None
        # self.pair_set = dict()
        # for i in self.target_indices:
        #     self.target_gradients[i] = None

    def set_backdoor_indices(self, target_indices, backdoor_indices, backdoor_X):
        self.target_indices = target_indices
        self.backdoor_indices = backdoor_indices
        self.backdoor_X = backdoor_X

    def receive_gradients(self, gradients):
        gradients = gradients.clone()
        # get the target gradient of samples labeled backdoor class
        for index, i in enumerate(self.indices):
            # if index.item() in self.target_indices:
            #     self.target_grad = gradients[i]
            #     break
            if i.item() in self.target_indices:
                self.target_gradients[i.item()] = gradients[index]

        # replace the gradient of backdoor samples with the target gradient
        for index, j in enumerate(self.indices):
            if j.item() in self.backdoor_indices:
                for i, v in self.pair_set.items():
                    if v == j.item():
                        target_grad = self.target_gradients[i]
                        if target_grad is not None:
                            gradients[index] = self.amplify_ratio * target_grad
                        break

        self.common_grad = gradients
        self._fit(self.X, self.components)

    def send_components(self):
        result = self._forward_computation(self.X)
        self.components = result
        send_result = result.clone()
        # send random latent representation for backdoor samples in VFL with model splitting
        with torch.no_grad():
            for index, i in enumerate(self.indices):
                if i.item() in self.target_indices:
                    if i.item() not in self.pair_set.keys():
                        j = self.backdoor_indices[random.randint(0, len(self.backdoor_indices)-1)]
                        self.pair_set[i.item()] = j
                    else:
                        j = self.pair_set[i.item()]
                    send_result[index] = self._forward_computation(torch.unsqueeze(self.backdoor_X[j], 0))

        return send_result

    def random_components(self, shape):
        variance = 1e-6
        result = torch.randn(shape)
        result = math.sqrt(variance)*result
        return result
