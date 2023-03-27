import torch
import torch.nn as nn

import torch.nn.functional as F

from vfl.defense.gradient_compression import GradientCompression
from vfl.defense.noisy_gradients import laplace_noise
from vfl.defense.norm_clip import norm_clip


class VFLActiveModel(object):
    """
    VFL active party
    """
    def __init__(self, bottom_model, args, top_model=None):
        super(VFLActiveModel, self).__init__()
        self.bottom_model = bottom_model
        self.is_debug = False

        self.classifier_criterion = nn.CrossEntropyLoss()
        self.parties_grad_component_list = []  # latent representations from passive parties
        self.X = None
        self.y = None
        self.bottom_y = None  # latent representation from local bottom model
        self.top_grads = None  # gradients of local bottom model
        self.parties_grad_list = []  # gradients for passive parties
        self.epoch = None  # current train epoch
        self.indices = None  # indices of current train samples

        self.top_model = top_model
        self.top_trainable = True if self.top_model is not None else False

        self.args = args

    def set_indices(self, indices):
        self.indices = indices

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_batch(self, X, y):
        self.X = X
        self.y = y

    def _fit(self, X, y):
        """
        compute gradients, and update local bottom model and top model

        :param X: features of active party
        :param y: labels
        """
        # get local latent representation
        self.bottom_y = self.bottom_model.forward(X)
        self.K_U = self.bottom_y.detach().requires_grad_()

        # compute gradients based on labels, including gradients for passive parties
        self._compute_common_gradient_and_loss(y)

        # update parameters of local bottom model and top model
        self._update_models(X, y)

    def predict(self, X, component_list):
        """
        get the final prediction

        :param X: feature of active party
        :param component_list: latent representations from passive parties
        :return: prediction label
        """
        # get local latent representation
        U = self.bottom_model.forward(X)

        # clip latent representation if using clip defense
        if 'norm_clip' in self.args and self.args['norm_clip']:
            for component in component_list:
                for value in component:
                    norm_clip(value, max_norm=self.args['clip_threshold'])
            for temp in U:
                norm_clip(temp, max_norm=self.args['clip_threshold'])

        # sum up latent representation in VFL without model splitting
        if not self.top_trainable:
            for comp in component_list:
                U = U + comp
        # use top model to predict in VFL with model splitting
        else:
            temp = torch.cat([U]+component_list, 1)
            U = self.top_model.forward(temp)

        result = F.softmax(U, dim=1)
        return result

    def receive_components(self, component_list):
        """
        receive latent representations from passive parties

        :param component_list: latent representations from passive parties
        """
        for party_component in component_list:
            self.parties_grad_component_list.append(party_component.detach().requires_grad_())

    def fit(self):
        """
        backward
        """
        self.parties_grad_list = []
        self._fit(self.X, self.y)
        self.parties_grad_component_list = []

    def _compute_common_gradient_and_loss(self, y):
        """
        compute loss and gradients, including gradients for passive parties

        :param y: label
        """
        # compute prediction
        U = self.K_U
        grad_comp_list = [self.K_U] + self.parties_grad_component_list
        if not self.top_trainable:
            for grad_comp in self.parties_grad_component_list:
                U = U + grad_comp
        else:
            temp = torch.cat(grad_comp_list, 1)
            U = self.top_model.forward(temp)

        # compute loss
        y = y.long().to(U.device)
        class_loss = self.classifier_criterion(U, y)

        # compute gradients
        if self.top_trainable:
            class_loss.backward(retain_graph=True)
            grad_list = [temp.grad for temp in grad_comp_list]
        else:
            grad_list = torch.autograd.grad(outputs=class_loss, inputs=grad_comp_list)
        # save gradients of local bottom model
        self.top_grads = grad_list[0]
        # save gradients for passive parties
        for index in range(0, len(self.parties_grad_component_list)):
            parties_grad = grad_list[index+1]
            self.parties_grad_list.append(parties_grad)

        # add noise to gradients for passive parties if using noisy gradient defense
        if 'noisy_gradients' in self.args and self.args['noisy_gradients']:
            self.top_grads = laplace_noise(self.top_grads, self.args['noise_scale'])
            for i, party_grad in enumerate(self.parties_grad_list):
                self.parties_grad_list[i] = laplace_noise(party_grad, self.args['noise_scale'])
        # compress gradient for all parties if using gradient compression defense
        elif 'gradient_compression' in self.args and self.args['gradient_compression']:
            gc = GradientCompression(gc_percent=self.args['gc_percent'])
            self.top_grads = gc.compression(self.top_grads)
            for i in range(0, len(self.parties_grad_list)):
                self.parties_grad_list[i] = gc.compression(self.parties_grad_list[i])

        self.loss = class_loss.item()*self.K_U.shape[0]

    def send_gradients(self):
        """
        send gradients to passive parties
        """
        return self.parties_grad_list

    def _update_models(self, X, y):
        """
        update parameters of local bottom model and top model

        :param X: features of active party
        :param y: invalid
        """
        # update parameters of local bottom model
        self.bottom_model.backward(X, self.bottom_y, self.top_grads)
        # update parameters of top model
        if self.top_trainable:
            self.top_model.backward_()

    def get_loss(self):
        return self.loss

    def save(self):
        """
        save model to local file
        """
        if self.top_trainable:
            self.top_model.save()
        self.bottom_model.save()

    def load(self):
        """
        load model from local file
        """
        if self.top_trainable:
            self.top_model.load()
        self.bottom_model.load()

    def set_train(self):
        """
        set train mode
        """
        if self.top_trainable:
            self.top_model.train()
        self.bottom_model.train()

    def set_eval(self):
        """
        set eval mode
        """
        if self.top_trainable:
            self.top_model.eval()
        self.bottom_model.eval()

    def scheduler_step(self):
        """
        adjust learning rate during training
        """
        if self.top_trainable and self.top_model.scheduler is not None:
            self.top_model.scheduler.step()
        if self.bottom_model.scheduler is not None:
            self.bottom_model.scheduler.step()

    def set_args(self, args):
        self.args = args

    def zero_grad(self):
        """
        clear gradients
        """
        if self.top_trainable:
            self.top_model.zero_grad()
        self.bottom_model.zero_grad()


class VFLPassiveModel(object):
    """
    VFL passive party
    """
    def __init__(self, bottom_model, id=None):
        super(VFLPassiveModel, self).__init__()
        self.bottom_model = bottom_model
        self.is_debug = False
        self.common_grad = None  # gradients
        # self.partial_common_grad = None
        self.X = None
        self.indices = None
        self.epoch = None
        self.y = None
        self.id = id  # id of passive party

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_batch(self, X, indices):
        self.X = X
        self.indices = indices

    def _forward_computation(self, X, model=None):
        """
        forward

        :param X: features of passive party
        :param model: invalid
        :return: latent representation of passive party
        """
        if model is None:
            A_U = self.bottom_model.forward(X)
        else:
            A_U = model.forward(X)
        self.y = A_U
        return A_U

    def _fit(self, X, y):
        """
        backward

        :param X: features of passive party
        :param y: latent representation of passive party
        """
        self.bottom_model.backward(X, y, self.common_grad, self.epoch)
        return

    def receive_gradients(self, gradients):
        """
        receive gradients from active party and update parameters of local bottom model

        :param gradients: gradients from active party
        """
        self.common_grad = gradients
        self._fit(self.X, self.y)

    def send_components(self):
        """
        send latent representation to active party
        """
        result = self._forward_computation(self.X)
        return result

    def predict(self, X):
        return self._forward_computation(X)

    def save(self):
        """
        save model to local file
        """
        self.bottom_model.save(id=self.id)

    def load(self, load_attack=False):
        """
        load model from local file

        :param load_attack: invalid
        """
        if load_attack:
            self.bottom_model.load(name='attack')
        else:
            self.bottom_model.load(id=self.id)

    def set_train(self):
        """
        set train mode
        """
        self.bottom_model.train()

    def set_eval(self):
        """
        set eval mode
        """
        self.bottom_model.eval()

    def scheduler_step(self):
        """
        adjust learning rate during training
        """
        if self.bottom_model.scheduler is not None:
            self.bottom_model.scheduler.step()

    def zero_grad(self):
        """
        clear gradients
        """
        self.bottom_model.zero_grad()
