# -*- coding: utf-8 -*-

"""
use Mix-Text to train inference head for text dataset, refers to "Label Inference Against Vertical Federated Learning"
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from common.utils import accuracy
from model.attack.lr_ba_top_model import lr_ba_top_model

total_steps = 0
flag = 0


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        probs_u = torch.softmax(outputs_u, dim=1)
        Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')
        Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                               * F.log_softmax(outputs_u, dim=1), dim=1) - self.args['lr_ba_top_margin'], min=0))
        return Lx, Lu, self.args['lr_ba_top_lambda_u'] * linear_rampup(epoch, rampup_length=self.args['lr_ba_top_epochs']), \
               Lu2, self.args['lr_ba_top_lambda_u_hinge'] * linear_rampup(epoch, rampup_length=self.args['lr_ba_top_epochs'])


def train_top_model_by_mix_text(train_loader, test_loader, labeled_loader, unlabeled_loader, bottom_model, args):
    param_dict = {
        'dataset': args['dataset'],
        'cuda': args['cuda'],
        'input_dim': bottom_model.output_dim,
        'output_dim': args['num_classes'],
        'pos': 'top',
        'lr': None
    }
    top_model = lr_ba_top_model(param_dict=param_dict, ema=False)
    optimizer = optim.Adam(top_model.parameters(), lr=args['lr_ba_top_lr'])

    train_acc = predict(loader=train_loader,
                        bottom_model=bottom_model,
                        top_model=top_model,
                        use_cuda=args['cuda'])
    test_acc = predict(loader=test_loader,
                       bottom_model=bottom_model,
                       top_model=top_model,
                       use_cuda=args['cuda'])
    logging.info('before mix_text train acc: {}'.format(train_acc))
    logging.info('before mix_text test acc: {}'.format(test_acc))

    criterion = SemiLoss(args)
    for epoch in range(args['lr_ba_top_epochs']):
        train(labeled_loader, unlabeled_loader, bottom_model, top_model, optimizer,
              criterion, epoch, args)

    train_acc = predict(loader=train_loader,
                        bottom_model=bottom_model,
                        top_model=top_model,
                        use_cuda=args['cuda'])
    test_acc = predict(loader=test_loader,
                       bottom_model=bottom_model,
                       top_model=top_model,
                       use_cuda=args['cuda'])
    logging.info('after mix_text train acc: {}'.format(train_acc))
    logging.info('after mix_text test acc: {}'.format(test_acc))

    return top_model


def train(labeled_loader, unlabeled_loader, bottom_model, top_model, optimizer, criterion, epoch, args):
    n_labels = args['num_classes']
    train_aug = args['lr_ba_train_aug']
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    bottom_model.eval()
    top_model.train()

    global total_steps
    global flag
    if flag == 0 and total_steps > args['lr_ba_top_temp_change']:
        print('Change T!')
        args['lr_ba_top_T'] = 0.9
        flag = 1

    for batch_idx in range(args['lr_ba_top_train_iteration']):
        total_steps += 1
        if not train_aug:
            try:
                data_zip, target_length_zip, _ = labeled_iter.next()
                inputs_x, _ = data_zip
                targets_x, inputs_x_length, _ = target_length_zip
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                data_zip, target_length_zip, _ = labeled_iter.next()
                inputs_x, _ = data_zip
                targets_x, inputs_x_length, _ = target_length_zip
        else:
            try:
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug), _ = labeled_iter.next()
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug), _ = labeled_iter.next()
        try:
            _, ((inputs_u, inputs_u2,  inputs_ori),
             (length_u, length_u2,  length_ori)), _ = unlabeled_iter.next()
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            _, ((inputs_u, inputs_u2, inputs_ori),
             (length_u, length_u2, length_ori)), _ = unlabeled_iter.next()

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_ori.size(0)
        targets_x = torch.zeros(batch_size, n_labels).scatter_(
            1, targets_x.view(-1, 1).long(), 1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_ori = inputs_ori.cuda()

        with torch.no_grad():
            # Predict labels for unlabeled data.
            outputs_u = top_model.forward(bottom_model.forward(inputs_u))
            outputs_u2 = top_model.forward(bottom_model.forward(inputs_u2))
            outputs_ori = top_model.forward(bottom_model.forward(inputs_ori))

            # Based on translation qualities, choose different weights here.
            # For AG News: German: 1, Russian: 0, ori: 1
            # For DBPedia: German: 1, Russian: 1, ori: 1
            # For IMDB: German: 0, Russian: 0, ori: 1
            # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
            p = (0 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2, dim=1) +
                 1 * torch.softmax(outputs_ori, dim=1)) / (1)
            # Do a sharpen here.
            pt = p**(1/args['lr_ba_top_T'])
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        mixed = 1
        if args['lr_ba_top_co']:
            mix_ = np.random.choice([0, 1], 1)[0]
        else:
            mix_ = 1

        if mix_ == 1:
            l = np.random.beta(args['lr_ba_top_alpha'], args['lr_ba_top_alpha'])
            if args['lr_ba_top_separate_mix']:
                l = l
            else:
                l = max(l, 1-l)
        else:
            l = 1

        mix_layer = np.random.choice(args['lr_ba_top_mix_layers_set'], 1)[0]
        mix_layer = mix_layer - 1

        if not train_aug:
            all_inputs = torch.cat(
                [inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)
            all_lengths = torch.cat(
                [inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)
        else:
            all_inputs = torch.cat(
                [inputs_x, inputs_x_aug, inputs_u, inputs_u2, inputs_ori], dim=0)
            all_lengths = torch.cat(
                [inputs_x_length, inputs_x_length, length_u, length_u2, length_ori], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)
        if args['lr_ba_top_separate_mix']:
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)
        else:
            idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
            idx2 = torch.arange(batch_size_2) + \
                all_inputs.size(0) - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]

        # Mix sentences' hidden representations
        logits = top_model.forward(bottom_model.forward(input_a, input_b, l, mix_layer))
        mixed_target = l * target_a + (1 - l) * target_b

        Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:], epoch+batch_idx/args['lr_ba_top_train_iteration'], mixed)
        if mix_ == 1:
            loss = Lx + w * Lu
        else:
            loss = Lx + w * Lu + w2 * Lu2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def predict(loader, bottom_model, top_model, use_cuda):
    # switch to evaluate mode
    bottom_model.eval()
    top_model.eval()
    y_predict = []
    y_true = []
    with torch.no_grad():
        for batch_idx, ((_, inputs), targets, _) in enumerate(loader):
            inputs = inputs.long()
            targets = targets[0].long()
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            temp_inputs = bottom_model.forward(inputs)
            outputs = top_model.forward(temp_inputs)
            # measure accuracy and record loss
            y_prob_preds = F.softmax(outputs, dim=1)
            y_true += targets.data.tolist()
            y_predict += y_prob_preds.tolist()
    acc = accuracy(y_true, y_predict, dataset='yahoo')
    return acc
