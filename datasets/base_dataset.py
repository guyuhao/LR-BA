# -*- coding: utf-8 -*-
import logging

from datasets.cifar_dataset import get_cifar_dataloader
from datasets.nus_wide_dataset import get_nus_wide_dataloader
from datasets.yahoo_dataset import get_yahoo_dataloader
from datasets.cinic_dataset import get_cinic_dataloader
from datasets.bhi_dataset import get_bhi_dataloader


def get_dataloader(args):
    """
    generate data loader according to dataset name

    :param args: configuration
    :return: data loader
    """
    if args['dataset'] == 'nus_wide':
        return get_nus_wide_dataloader(args)
    elif 'cifar' in args['dataset']:
        return get_cifar_dataloader(args)
    elif args['dataset'] == 'yahoo':
        return get_yahoo_dataloader(args)
    elif args['dataset'] == 'cinic':
        return get_cinic_dataloader(args)
    elif args['dataset'] == 'bhi':
        return get_bhi_dataloader(args)


def get_backdoor_target_index(train_loader, backdoor_indices, args):
    """
    get index of a normal input labeled backdoor class in training dataset, used for gradient-replacement backdoor

    :param train_loader: loader of training dataset
    :param backdoor_indices: indices of backdoor samples
    :param args: configuration
    :return: index of a normal input labeled backdoor class
    """
    for (_, _), labels, indices in train_loader:
        for label, index in zip(labels, indices):
            if label == args['backdoor_label'] and index not in backdoor_indices:
                logging.info('backdoor target index: {}'.format(index))
                return index.item()
    return None


def get_num_classes(dataset):
    """
    get classes number of the target dataset

    :param str dataset: target dataset name
    :return: classes number of the target dataset
    """
    data_dict = {
        'nus_wide': 5,
        'cifar10': 10,
        'cifar100': 100,
        'yahoo': 10,
        'cinic': 10,
        'bhi': 2
    }
    return data_dict[dataset]


