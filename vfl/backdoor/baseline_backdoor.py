#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gyh 
@File    ：baseline_backdoor.py
@Author  ：Gu Yuhao
@Date    ：2022/5/19 下午7:47 

"""
import copy
import logging
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data

from common.utils import accuracy
from vfl.backdoor.mix_match import train_top_model_by_mix_match
from vfl.backdoor.mix_text import train_top_model_by_mix_text


def train_top_model(train_loader, test_loader, labeled_loader, unlabeled_loader, bottom_model, args):
    """
    train the inference head, refers to "Label Inference Attacks Against Federated Learning"

    :param train_loader: loader of normal train dataset
    :param test_loader: loader of normal test dataset
    :param labeled_loader: loader of labeled normal train dataset
    :param unlabeled_loader: loader of unlabeled normal train dataset
    :param bottom_model: bottom model of the attacker
    :param args: configuration
    :return: inference head
    """
    for param in bottom_model.parameters():
        param.requires_grad = False

    if args['dataset'] != 'yahoo':
        # use Mix-Match to train the inference head for feature and image dataset
        return train_top_model_by_mix_match(train_loader=train_loader,
                                            test_loader=test_loader,
                                            labeled_loader=labeled_loader,
                                            unlabeled_loader=unlabeled_loader,
                                            bottom_model=bottom_model,
                                            args=args)
    else:
        # use Mix-Text to train the inference head for text dataset
        return train_top_model_by_mix_text(train_loader=train_loader,
                                           test_loader=test_loader,
                                           labeled_loader=labeled_loader,
                                           unlabeled_loader=unlabeled_loader,
                                           bottom_model=bottom_model,
                                           args=args)


def finetune_bottom_model(old_bottom_model, top_model, labeled_loader, train_loader, backdoor_indices, args):
    """
    fine-tune to get the malicious bottom model

    :param old_bottom_model: pre-trained bottom model of the attacker
    :param backdoor_output_list: backdoor latent representation
    :param train_loader: loader of normal train dataset
    :param backdoor_indices: indices of backdoor samples in normal train dataset
    :param args: configuration
    :return: malicious bottom model
    """
    for param in old_bottom_model.parameters():
        param.requires_grad = True

    for param in top_model.parameters():
        param.requires_grad = False

    X, y = torch.tensor([]), torch.tensor([])
    backdoor_X, backdoor_y = torch.tensor([]), torch.tensor([])

    bottom_model = copy.deepcopy(old_bottom_model)
    # generate normal inputs of fine-tune dataset
    X, y = torch.tensor([]), torch.tensor([])
    backdoor_X, backdoor_y = torch.tensor([]), torch.tensor([])

    for _, (inputs, _, indices) in enumerate(train_loader):
        if args['dataset'] != 'bhi':
            _, Xb_inputs = inputs
        else:
            Xb_inputs = inputs[:, 1]
        temp_indices = []
        temp_y = bottom_model.forward(Xb_inputs).detach()
        for i, index in enumerate(indices):
            if index.item() in backdoor_indices:
                backdoor_X = torch.cat([backdoor_X, Xb_inputs[i:i + 1].float()], dim=0)
                temp_indices.append(i)
        normal_indices = np.setdiff1d(np.arange(len(Xb_inputs)), temp_indices)
        Xb_inputs = Xb_inputs[normal_indices]
        temp_y = temp_y[normal_indices]
        X = torch.cat([X, Xb_inputs.float()], dim=0)
        y = torch.cat([y, temp_y.cpu()], dim=0)
    ds = data.TensorDataset(torch.tensor(np.array(X)), torch.tensor(np.array(y)))
    dl = data.DataLoader(dataset=ds,
                         batch_size=args['target_batch_size'],
                         shuffle=True,
                         drop_last=False)
    # generate backdoor inputs of fine-tune dataset, whose label is all backdoor representation
    backdoor_y = torch.full((backdoor_X.shape[0],), fill_value=args['backdoor_label'])
    backdoor_ds = data.TensorDataset(torch.tensor(np.array(backdoor_X)), backdoor_y)
    backdoor_dl = data.DataLoader(dataset=backdoor_ds,
                                  batch_size=args['target_batch_size'],
                                  shuffle=True,
                                  drop_last=False)

    bottom_model.train()
    top_model.eval()

    if args['dataset'] != 'yahoo':
        optimizer = optim.Adam(bottom_model.parameters(), lr=args['lr_ba_finetune_lr'])
    else:
        parameters = [{"params": bottom_model.bert.parameters(), "lr": 5e-6},
                      {"params": bottom_model.linear.parameters(), "lr": 5e-4}]
        optimizer = optim.Adam(parameters, lr=args['lr_ba_finetune_lr'])

    scheduler = None
    backdoor_criterion = nn.CrossEntropyLoss()
    normal_criterion = nn.MSELoss()
    for ep in range(args['lr_ba_finetune_epochs']):
        # use the same size of normal and backdoor inputs in one batch
        # the size of backdoor inputs is smaller than normal inputs, so use cycle
        for batch_idx, temp_data in enumerate(zip(dl, cycle(backdoor_dl))):
            (X, target_y), (backdoor_X, backdoor_target_y) = temp_data
            if args['dataset'] == 'yahoo':
                X, backdoor_X = X.long(), backdoor_X.long()
            if args['cuda']:
                X, target_y = X.cuda(), target_y.float().cuda()
                backdoor_X, backdoor_target_y = backdoor_X.cuda(), backdoor_target_y.long().cuda()
            backdoor_predict_y = top_model.forward(bottom_model.forward(backdoor_X))
            backdoor_loss = backdoor_criterion(backdoor_predict_y, backdoor_target_y)
            normal_predict_y = bottom_model.forward(X)
            normal_loss = normal_criterion(normal_predict_y, target_y)
            loss = normal_loss + backdoor_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
    return bottom_model


def poison_predict(top_model, bottom_model, test_loader, dataset, args, top_k=1, num_classes=10):
    """
    evaluate LR-BA using the backdoor latent representation directly without fine-tuning bottom model, only for debugging

    :param vfl: vfl
    :param target_list: backdoor latent representation
    :param test_loader: loader of backdoor test dataset
    :param dataset: dataset name
    :param args: configuration
    :param is_attack: whether to evaluate attack accuracy
    :param top_k: top-k accuracy
    :param num_classes: number of dataset classes
    :return: attack accuracy
    """
    top_model.eval()
    bottom_model.eval()

    y_predict = []
    y_true = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(test_loader):
            party_X_test_dict = dict()
            if dataset != 'bhi':
                Xa_inputs, Xb_inputs = inputs
                if dataset == 'yahoo':
                    Xa_inputs = Xa_inputs.long()
                    Xb_inputs = Xb_inputs.long()
                    targets = targets[0].long()
                party_X_test_dict[0] = Xb_inputs
            else:
                Xa_inputs = inputs[:, 0]
                for i in range(args['n_passive_party']):
                    party_X_test_dict[i] = inputs[:, i + 1:i + 2].squeeze(1)
                Xb_inputs = party_X_test_dict[0]
            y_prob_preds = F.softmax(top_model.forward(bottom_model.forward(Xb_inputs)))
            y_true += targets.data.tolist()
            y_predict += y_prob_preds.tolist()
        acc = accuracy(y_true, y_predict, top_k=top_k, num_classes=num_classes, is_attack=True, dataset=dataset)
    return acc


def baseline_backdoor(train_loader, test_loader, backdoor_indices,
                      labeled_loader, unlabeled_loader, vfl, args,
                      backdoor_train_loader=None, backdoor_test_loader=None, lr_ba_top_model=None):
    """
    implementation of LR-BA

    :param train_loader: loader of normal train dataset
    :param test_loader: loader of normal test dataset
    :param backdoor_indices: indices of backdoor samples in normal train dataset
    :param labeled_loader: loader of labeled samples in normal train dataset
    :param unlabeled_loader: loader of unlabeled samples in normal train dataset
    :param vfl: vfl
    :param args: configuration
    :param backdoor_train_loader: loader of backdoor train dataset
    :param backdoor_test_loader: loader of backdoor test dataset
    :param lr_ba_top_model: inference head, not training if provided
    :param add_from_unlabeled_tag: whether to initialize from unlabeled samples
    :param filter_by_shap_tag: invalid
    :param get_error_features_tag: invalid
    :param random_tag: whether to randomly initialize backdoor latent representation
    :return: tuple containing: malicious bottom model and inference head
    """
    new_bottom_model = None
    bottom_model = vfl.party_dict[0].bottom_model  # bottom model of the attacker

    load = False

    if not load:
        # train the inference head if it isn't provided
        if lr_ba_top_model is None:
            lr_ba_top_model = train_top_model(train_loader=train_loader,
                                            test_loader=test_loader,
                                            labeled_loader=labeled_loader,
                                            unlabeled_loader=unlabeled_loader,
                                            bottom_model=bottom_model,
                                            args=args)

        if 'debug' in args and not args['debug']:
            # fine-tune to get the malicious bottom model
            new_bottom_model = finetune_bottom_model(old_bottom_model=bottom_model,
                                                     top_model=lr_ba_top_model,
                                                     labeled_loader=labeled_loader,
                                                     train_loader=backdoor_train_loader,
                                                     backdoor_indices=backdoor_indices,
                                                     args=args)

    acc = poison_predict(
        lr_ba_top_model, new_bottom_model, backdoor_test_loader, args['dataset'], args,
        top_k=1,
        num_classes=args['num_classes']
    )
    logging.info("--- debug baseline attack acc: {0}".format(acc))

    return new_bottom_model

