# -*- coding: utf-8 -*-

"""
implementation of LR-BA
"""
import copy
import logging
from itertools import cycle

import numpy as np
import shap
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numpy import median, mean
from torch import nn, optim
from torch.utils import data

from common.utils import accuracy, print_running_time
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


def add_from_unlabeled(bottom_model, top_model, args, unlabeled_loader, train_size, top_k=1):
    """
    get latent representations of backdoor label from unlabeled samples

    :param bottom_model: bottom model of the attacker
    :param top_model: inference head
    :param args: configuration
    :param unlabeled_loader: loader of unlabeled normal train dataset
    :param int train_size: size of normal train dataset
    :param int top_k: top-k label, 5 for CIFAR-100 and 1 for other datasets
    :return: latent representations of backdoor label from unlabeled samples
    """
    backdoor_label = args['backdoor_label']
    y_list = []
    predict_list = []

    # select unlabeled samples of backdoor label
    if args['dataset'] != 'yahoo':
        for inputs, _, _ in unlabeled_loader:
            if args['dataset'] != 'bhi':
                _, X = inputs
            else:
                X = inputs[:, 1]
            y = F.softmax(top_model.forward(bottom_model.forward(X)), dim=1)
            predict_labels = torch.topk(y.data, k=top_k, dim=1)[1]
            target_indices = []
            for i, predict_label in enumerate(predict_labels):
                if backdoor_label in predict_label.tolist():
                    target_indices.append(i)
            if len(target_indices) > 0:
                temp = torch.unsqueeze(bottom_model.forward(X[target_indices]), 1)
                y_list += temp.tolist()
                predict_list += y[target_indices].tolist()
    else:
        for _, ((X, _, _), (_, _, _)), _ in unlabeled_loader:
            X = X.long()
            y = F.softmax(top_model.forward(bottom_model.forward(X)), dim=1)
            predict_labels = torch.max(y.data, 1)[1]
            target_indices = []
            for i, predict_label in enumerate(predict_labels):
                if predict_label.item() == backdoor_label:
                    target_indices.append(i)
            if len(target_indices) > 0:
                temp = torch.unsqueeze(bottom_model.forward(X[target_indices]), 1)
                y_list += temp.tolist()
                predict_list += y[target_indices].tolist()
    logging.info('unlabeled y_list size: {}'.format(len(y_list)))

    # order latent representations according to the prediction probability of the backdoor label
    # select top M latent representations as result
    if len(predict_list) > 0:
        torch_predict = torch.tensor(predict_list)
        temp, sorted_indices = torch.sort(torch_predict, descending=True, dim=0)
        target_sorted_indices = sorted_indices[:, args['backdoor_label']]
        np_y_list = np.array(y_list)
        target_y = np_y_list[target_sorted_indices.cpu().numpy().tolist()]
        if 'backdoor_label_size' in args.keys() and args['backdoor_label_size'] > 0:
            # select with prior knowledge
            filter_size = int(args['backdoor_label_size']-args['train_label_size']/args['num_classes'])
        else:
            # select without prior knowledge, assuming classes distribute uniformly in train dataset
            filter_size = int((train_size-args['train_label_size']) / args['num_classes'])
        y_list = target_y[:filter_size].tolist()
    return y_list


def filter_by_shap(labeled_loader, bottom_model, top_model, args, y_list):
    """
    filter latent representations by shap value, however it doesn't work
    """
    background = None
    for X, _, _ in labeled_loader:
        if args['dataset'] != 'bhi':
            _, Xb = X
        else:
            Xb = X[:, 1]
        temp = bottom_model.forward(Xb)
        if background is None:
            background = temp
        else:
            background = torch.cat([background, temp], 0)
    background = torch.split(background, 100, dim=0)[0]
    e = shap.DeepExplainer(top_model, background)
    shap_values = e.shap_values(y_list.squeeze(1))

    np_shap_values = np.array(shap_values)
    temp_indices = []
    for i, y in enumerate(y_list):
        shap_value = np_shap_values[:, i, :]
        temp_list = []
        for j in range(args['num_classes']):
            temp_list.append(np.sum(shap_value[j]))
        if max(temp_list) == temp_list[args['backdoor_label']]:
            temp_indices.append(i)
    logging.info('final indices size: {}'.format(len(temp_indices)))
    return temp_indices, np_shap_values


def get_error_features(temp_indices, args, np_shap_values=None,
                       labeled_loader=None, bottom_model=None, top_model=None, y_list=None):
    """
    find dimensions in backdoor latent representation that shouldn't be optimized by shap value, however it doesn't work
    """
    if np_shap_values is None:
        background = None
        for X, _, _ in labeled_loader:
            if args['dataset'] != 'bhi':
                _, Xb = X
            else:
                Xb = X[:, 1]
            temp = bottom_model.forward(Xb)
            if background is None:
                background = temp
            else:
                background = torch.cat([background, temp], 0)
        background = torch.split(background, 100, dim=0)[0]
        e = shap.DeepExplainer(top_model, background)
        shap_values = e.shap_values(y_list.squeeze(1))
        np_shap_values = np.array(shap_values)

    target_shap_values = np_shap_values[args['backdoor_label'], temp_indices]
    error_features = []
    feature_order = np.argsort(np.sum(np.abs(target_shap_values), axis=0))
    for i in range(target_shap_values.shape[1]):
        temp = target_shap_values[:, i]
        middle = median(temp)
        if middle < 0:
            if i in feature_order[int(len(feature_order)/2):]:
                error_features.append(i)
    logging.info('error features: {}'.format(error_features))
    return error_features


def get_original_backdoor_output(labeled_loader, bottom_model, top_model, args, unlabeled_loader, top_k=1,
                                 add_from_unlabeled_tag=True, filter_by_shap_tag=True, get_error_features_tag=True,
                                 random_tag=False):
    """
    generate initialization of backdoor latent representation

    :param labeled_loader: loader of labeled samples in normal train dataset
    :param bottom_model: bottom model of the attacker
    :param top_model: inference head
    :param args: configuration
    :param unlabeled_loader: loader of unlabeled samples in normal train dataset
    :param int top_k: top-k label
    :param add_from_unlabeled_tag: whether to initialize from unlabeled samples
    :param filter_by_shap_tag: invalid
    :param get_error_features_tag: invalid
    :param random_tag: whether to randomly initialize latent representation
    :return: tuple containing initialization of backdoor latent representation and _
    """
    bottom_model.eval()
    top_model.eval()
    backdoor_label = args['backdoor_label']

    # randomly initialize latent representation
    if random_tag:
        result = torch.randn(size=(1, top_model.input_dim))
        if args['cuda']:
            result = result.cuda()
        return result, []

    train_size = len(labeled_loader.dataset) + len(unlabeled_loader.dataset)
    y_list = []
    # get latent representations of backdoor label from labeled samples
    for _, (inputs, label, _) in enumerate(labeled_loader.dataset):
        if args['dataset'] != 'bhi':
            _, X = inputs
            if args['dataset'] == 'yahoo':
                label = label[0]
        else:
            X = inputs[1]
        if label == backdoor_label:
            if args['dataset'] == 'yahoo':
                X = X.long()
            y = bottom_model.forward(torch.unsqueeze(X, 0))
            y_list.append(y.tolist())
    logging.info('y_list size: {}'.format(len(y_list)))

    # get latent representations of backdoor label from unlabeled samples
    if add_from_unlabeled_tag:
        y_list += add_from_unlabeled(bottom_model, top_model, args, unlabeled_loader, train_size=train_size, top_k=top_k)

    logging.info('y_list size: {}'.format(len(y_list)))
    y_list = torch.tensor(y_list)
    temp_indices = list(range(0, len(y_list)))
    error_features = []

    # calculate average as the initialization of backdoor latent representation
    result = torch.mean(y_list[temp_indices], dim=0)
    if result.shape[0] != 1:
        result = torch.unsqueeze(result, 0)
    if args['cuda']:
        result = result.cuda()
    return result, error_features


def generate_backdoor_output(labeled_loader, bottom_model, top_model, args, unlabeled_loader=None, top_k=1,
                             add_from_unlabeled_tag=True, filter_by_shap_tag=True, get_error_features_tag=True,
                             random_tag=False):
    """
    generate backdoor latent representation

    :param labeled_loader: loader of labeled samples in normal train dataset
    :param bottom_model: bottom model of the attacker
    :param top_model: inference head
    :param args: configuration
    :param unlabeled_loader: loader of unlabeled samples in normal train dataset
    :param top_k: top-k label
    :param add_from_unlabeled_tag: whether to initialize from unlabeled samples
    :param filter_by_shap_tag: invalid
    :param get_error_features_tag: invalid
    :param random_tag: whether to randomly initialize latent representation
    :return: backdoor latent representation
    """
    bottom_model.eval()
    top_model.eval()
    for param in top_model.parameters():
        param.requires_grad = False

    # get initialization of backdoor latent representation
    result, error_features = get_original_backdoor_output(
        labeled_loader=labeled_loader,
        bottom_model=bottom_model,
        top_model=top_model,
        args=args,
        unlabeled_loader=unlabeled_loader,
        top_k=top_k,
        add_from_unlabeled_tag=add_from_unlabeled_tag,
        filter_by_shap_tag=filter_by_shap_tag,
        get_error_features_tag=get_error_features_tag,
        random_tag=random_tag
    )

    logging.info("--- start backdoor output: {0}".format(result.data))
    criterion = nn.CrossEntropyLoss()
    distance_criterion = nn.MSELoss()
    old_result = result.clone()
    old_result.requires_grad = False
    logging.info('old norm: {}'.format(torch.norm(old_result, p=2)))
    optim_list = []
    result_list = result.split(1, dim=1)
    for i, temp in enumerate(result_list):
        if i in error_features:
            temp.requires_grad = False
        else:
            temp.requires_grad = True
            optim_list.append(temp)
    optimizer = optim.Adam(optim_list, lr=args['lr_ba_generate_lr'])
    result_data_list = []
    multi_epochs = False
    if isinstance(args['lr_ba_generate_epochs'], list):
        max_epochs = max(args['lr_ba_generate_epochs'])
        multi_epochs = True
    else:
        max_epochs = args['lr_ba_generate_epochs']
    if multi_epochs and 0 in args['lr_ba_generate_epochs']:
        result_data_list.append(result.clone())
    # optimize backdoor latent representation using Adam and cross entropy loss
    for epoch in range(0, max_epochs):
        result = torch.cat(result_list, dim=1)
        predict_y = top_model.forward(result)
        target_y = torch.tensor([args['backdoor_label']]).long()
        if args['cuda']:
            target_y = target_y.cuda()
        loss = criterion(predict_y, target_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if not multi_epochs:
            if epoch == 0 or epoch == max_epochs-1:
                logging.info("--- generate epoch: {0}, loss: {1}, generate y: {2}".format(epoch, loss.data,
                                                                                          F.softmax(predict_y, dim=1)))
                if epoch == max_epochs - 1:
                    result_data_list.append(result.clone())
                    logging.info("--- epoch: {}, backdoor output: {}".format(epoch, result.data))
        if multi_epochs and (epoch+1) in args['lr_ba_generate_epochs']:
            logging.info("--- generate epoch: {0}, loss: {1}, generate y: {2}".format(epoch, loss.data,
                                                                                      F.softmax(predict_y, dim=1)))
            result_data_list.append(result.clone())
            logging.info("--- epoch: {}, backdoor output: {}".format(epoch, result.data))

    logging.info('new norm: {}'.format(torch.norm(result, p=2)))
    # return result.data
    return result_data_list


def finetune_bottom_model(old_bottom_model, backdoor_output_list, train_loader, backdoor_indices, args):
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
    # old_bottom_model.eval()
    bottom_model_list = []
    X, y = torch.tensor([]), torch.tensor([])
    backdoor_X, backdoor_y = torch.tensor([]), torch.tensor([])
    for backdoor_output in backdoor_output_list:
        bottom_model = copy.deepcopy(old_bottom_model)
        bottom_model.eval()
        # generate normal inputs of fine-tune dataset
        if len(X) == 0:
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
                        # backdoor_y = torch.cat([backdoor_y, backdoor_output.cpu()], dim=0)
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
        backdoor_y = backdoor_output.cpu().repeat(backdoor_X.shape[0], 1).detach()
        backdoor_ds = data.TensorDataset(torch.tensor(np.array(backdoor_X)), torch.tensor(np.array(backdoor_y)))
        backdoor_dl = data.DataLoader(dataset=backdoor_ds,
                                      batch_size=args['target_batch_size'],
                                      shuffle=True,
                                      drop_last=False)

        bottom_model.train()
        if args['dataset'] != 'yahoo':
            optimizer = optim.Adam(bottom_model.parameters(), lr=args['lr_ba_finetune_lr'])
        else:
            parameters = [{"params": bottom_model.bert.parameters(), "lr": 5e-6},
                          {"params": bottom_model.linear.parameters(), "lr": 5e-4}]
            optimizer = optim.Adam(parameters, lr=args['lr_ba_finetune_lr'])

        scheduler = None
        criterion = nn.MSELoss()
        for ep in range(args['lr_ba_finetune_epochs']):
            loss_list = []
            # use the same size of normal and backdoor inputs in one batch
            # the size of backdoor inputs is smaller than normal inputs, so use cycle
            for batch_idx, temp_data in enumerate(zip(dl, cycle(backdoor_dl))):
                (X, target_y), (backdoor_X, backdoor_target_y) = temp_data
                if args['dataset'] == 'yahoo':
                    X, backdoor_X = X.long(), backdoor_X.long()
                X = torch.cat([X, backdoor_X], dim=0)
                target_y = torch.cat([target_y, backdoor_target_y], dim=0)
                if args['cuda']:
                    X, target_y = X.cuda(), target_y.float().cuda()
                predict_y = bottom_model.forward(X)
                loss = criterion(predict_y, target_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
        bottom_model_list.append(bottom_model)
    # return bottom_model
    return bottom_model_list


def poison_predict(
        vfl, target_list, test_loader, dataset, args, is_attack=True, top_k=1, num_classes=10):
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
    vfl.set_eval()

    y_predict = []
    y_true = []
    acc_list = []

    with torch.no_grad():
        for target in target_list:
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
                y_prob_preds = vfl.predict(Xa_inputs, party_X_test_dict, target.repeat(Xa_inputs.shape[0], 1))
                if is_attack:
                    targets = torch.full_like(targets, args['backdoor_label'])
                y_true += targets.data.tolist()
                y_predict += y_prob_preds.tolist()
            acc = accuracy(y_true, y_predict, top_k=top_k, num_classes=num_classes, is_attack=True, dataset=dataset)
            acc_list.append(acc)
    return acc_list


def lr_ba_backdoor(train_loader, test_loader, backdoor_indices,
                 labeled_loader, unlabeled_loader, vfl, args,
                 backdoor_train_loader=None, backdoor_test_loader=None,
                 lr_ba_top_model=None, add_from_unlabeled_tag=True, filter_by_shap_tag=True, get_error_features_tag=True,
                 random_tag=False):
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
    top_k = 1
    if args['dataset'] == 'cifar100':
        top_k = 5

    new_bottom_model_list = []
    bottom_model = vfl.party_dict[0].bottom_model  # bottom model of the attacker

    load = False

    if not load:
        # record the start time before model completion if evaluating execution time
        if args['time']:
            start_time = print_running_time(None, None)
        # train the inference head if it isn't provided
        if lr_ba_top_model is None:
            lr_ba_top_model = train_top_model(train_loader=train_loader,
                                            test_loader=test_loader,
                                            labeled_loader=labeled_loader,
                                            unlabeled_loader=unlabeled_loader,
                                            bottom_model=bottom_model,
                                            args=args)
        # print execution time of model completion
        if args['time']:
            start_time = print_running_time('model completion', start_time)

        # generate backdoor latent representation
        backdoor_output_list = generate_backdoor_output(labeled_loader=labeled_loader,
                                                        bottom_model=bottom_model,
                                                        top_model=lr_ba_top_model,
                                                        args=args,
                                                        unlabeled_loader=unlabeled_loader,
                                                        top_k=top_k,
                                                        add_from_unlabeled_tag=add_from_unlabeled_tag,
                                                        filter_by_shap_tag=filter_by_shap_tag,
                                                        get_error_features_tag=get_error_features_tag,
                                                        random_tag=random_tag)
        # print execution time of generate backdoor latent representation generation
        if args['time']:
            start_time = print_running_time('backdoor latent representation generation', start_time)

        # debug
        # evaluate LR-BA using backdoor latent representation directly without fine-tuning bottom model
        acc_list = poison_predict(
            vfl,
            backdoor_output_list, backdoor_test_loader, args['dataset'], args,
            top_k=top_k,
            num_classes=args['num_classes']
        )
        logging.info("--- debug LR-BA attack acc: {0}".format(acc_list))

        if 'debug' in args and not args['debug']:
            # record the start time before model fine-tuning if evaluating execution time
            if args['time']:
                start_time = print_running_time(None, None)
            # fine-tune to get the malicious bottom model
            new_bottom_model_list = finetune_bottom_model(old_bottom_model=bottom_model,
                                                          backdoor_output_list=backdoor_output_list,
                                                          train_loader=backdoor_train_loader,
                                                          backdoor_indices=backdoor_indices,
                                                          args=args)
            # print execution time of model fine-tuning
            if args['time']:
                start_time = print_running_time('model fine-tuning', start_time)
    return new_bottom_model_list, lr_ba_top_model


def lr_ba_backdoor_for_representation(train_loader, test_loader,
                                    labeled_loader, unlabeled_loader, vfl, args,
                                    lr_ba_top_model=None, add_from_unlabeled_tag=True, filter_by_shap_tag=True, get_error_features_tag=True,
                                    random_tag=False):
    """
    obtain backdoor latent representation generated during LR-BA, only support two-party VFL

    :param train_loader: loader of normal train dataset
    :param test_loader: loader of normal test dataset
    :param labeled_loader: loader of labeled samples in normal train dataset
    :param unlabeled_loader: loader of unlabeled samples in normal train dataset
    :param vfl: vfl
    :param args: configuration
    :param lr_ba_top_model: inference head, not training if provided
    :param add_from_unlabeled_tag: whether to initialize from unlabeled samples
    :param filter_by_shap_tag: invalid
    :param get_error_features_tag: invalid
    :param random_tag: whether to randomly initialize backdoor latent representation
    :return: tuple containing: malicious bottom model and backdoor latent representation
    """
    top_k = 1
    if args['dataset'] == 'cifar100':
        top_k = 5

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
        # generate backdoor latent representation
        backdoor_output_list = generate_backdoor_output(labeled_loader=labeled_loader,
                                                        bottom_model=bottom_model,
                                                        top_model=lr_ba_top_model,
                                                        args=args,
                                                        unlabeled_loader=unlabeled_loader,
                                                        top_k=top_k,
                                                        add_from_unlabeled_tag=add_from_unlabeled_tag,
                                                        filter_by_shap_tag=filter_by_shap_tag,
                                                        get_error_features_tag=get_error_features_tag,
                                                        random_tag=random_tag)
    return lr_ba_top_model, backdoor_output_list[0]
