# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from common.utils import accuracy
from model.attack.lr_ba_top_model import lr_ba_top_model

"""
use Mix-Match to train inference head for feature and image dataset, refers to "Label Inference Against Vertical Federated Learning"
"""


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __init__(self, epochs):
        self.epochs = epochs

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        lambda_u = 50
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, lambda_u * linear_rampup(epoch, self.epochs)


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.type(torch.float)
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param = param.type(torch.float)
            param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


# 参考MixMatch
def mix_match_train(labeled_loader, unlabeled_loader, bottom_model, top_model,
                    criterion, optimizer, ema_optimizer, epoch, args):
    use_cuda = args['cuda']
    num_classes = args['num_classes']
    T = args['lr_ba_top_T']
    alpha = args['lr_ba_top_alpha']

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    bottom_model.eval()
    top_model.train()

    for batch_idx in range(args['lr_ba_top_train_iteration']):
        try:
            inputs, targets_x, _ = labeled_iter.next()
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            inputs, targets_x, _ = labeled_iter.next()
        if args['dataset'] != 'bhi':
            _, inputs_x = inputs
        else:
            inputs_x = inputs[:, 1]

        try:
            inputs, _, _ = unlabeled_iter.next()
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            inputs, _, _ = unlabeled_iter.next()
        if args['dataset'] != 'bhi':
            _, inputs_u = inputs
        else:
            inputs_u = inputs[:, 1]

        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_classes).scatter_(1, targets_x.view(-1, 1).long(), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            temp = bottom_model.forward(inputs_u)
            outputs_u = top_model.forward(temp)
            p = torch.softmax(outputs_u, dim=1)
            pt = p**(1/T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        l = np.random.beta(alpha, alpha)
        l = max(l, 1-l)
        idx = torch.randperm(all_inputs.shape[0])

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [top_model.forward(bottom_model.forward(mixed_input[0]))]
        for input in mixed_input[1:]:
            logits.append(top_model.forward(bottom_model.forward(input)))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x,
                              mixed_target[:batch_size],
                              logits_u,
                              mixed_target[batch_size:],
                              epoch+batch_idx/args['lr_ba_top_train_iteration'])

        loss = Lx + w * Lu

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
    return


def train_top_model_by_mix_match(train_loader, test_loader, labeled_loader, unlabeled_loader, bottom_model, args):
    param_dict = {
        'dataset': args['dataset'],
        'cuda': args['cuda'],
        'input_dim': bottom_model.output_dim,
        'output_dim': args['num_classes'],
        'pos': 'top',
        'lr': None
    }
    top_model = lr_ba_top_model(param_dict=param_dict, ema=False)
    ema_top_model = lr_ba_top_model(param_dict=param_dict, ema=True)

    top_k = 1
    if args['dataset'] == 'cifar100':
        top_k = 5

    if not args['time']:
        train_acc = predict(loader=train_loader,
                            bottom_model=bottom_model,
                            top_model=top_model,
                            use_cuda=args['cuda'],
                            top_k=top_k,
                            num_classes=args['num_classes'],
                            dataset=args['dataset'])
        test_acc = predict(loader=test_loader,
                           bottom_model=bottom_model,
                           top_model=top_model,
                           use_cuda=args['cuda'],
                           top_k=top_k,
                           num_classes=args['num_classes'],
                           dataset=args['dataset'])
        logging.info('before mix_match train acc: {}'.format(train_acc))
        logging.info('before mix_match test acc: {}'.format(test_acc))

    criterion = SemiLoss(args['lr_ba_top_epochs'])
    optimizer = optim.Adam(top_model.parameters(), lr=args['lr_ba_top_lr'])
    ema_optimizer = WeightEMA(top_model, ema_top_model, lr=args['lr_ba_top_lr'], alpha=args['lr_ba_ema_decay'])
    for epoch in range(0, args['lr_ba_top_epochs']):
        mix_match_train(labeled_loader=labeled_loader,
                        unlabeled_loader=unlabeled_loader,
                        bottom_model=bottom_model,
                        top_model=top_model,
                        epoch=epoch,
                        criterion=criterion,
                        optimizer=optimizer,
                        ema_optimizer=ema_optimizer,
                        args=args)
    # not evaluate the performance of the inference head if evaluating execution time
    if not args['time']:
        train_acc = predict(loader=train_loader,
                            bottom_model=bottom_model,
                            # top_model=top_model,
                            top_model=ema_top_model,
                            use_cuda=args['cuda'],
                            num_classes=args['num_classes'],
                            top_k=top_k,
                            dataset=args['dataset'])
        test_acc = predict(loader=test_loader,
                           bottom_model=bottom_model,
                           # top_model=top_model,
                           top_model=ema_top_model,
                           use_cuda=args['cuda'],
                           num_classes=args['num_classes'],
                           top_k=top_k,
                           dataset=args['dataset'])
        logging.info('epoch: {}. train acc: {}'.format(epoch, train_acc))
        logging.info('epoch: {}. test acc: {}'.format(epoch, test_acc))
    if args['save_model']:
        top_model.save(name='lr_ba_top')
        ema_top_model.save(name='lr_ba_ema_top')
    return ema_top_model


def predict(loader, bottom_model, top_model, use_cuda, num_classes, dataset, top_k=1):
    # switch to evaluate mode
    bottom_model.eval()
    top_model.eval()
    y_predict = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (batch_inputs, targets, _) in enumerate(loader):
            if dataset != 'bhi':
                _, inputs = batch_inputs
            else:
                inputs = batch_inputs[:, 1]
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            temp_inputs = bottom_model.forward(inputs)
            outputs = top_model.forward(temp_inputs)
            # measure accuracy and record loss
            y_prob_preds = F.softmax(outputs, dim=1)
            y_true += targets.data.tolist()
            y_predict += y_prob_preds.tolist()
    acc = accuracy(y_true, y_predict, top_k=top_k, num_classes=num_classes, dataset=dataset)
    return acc
