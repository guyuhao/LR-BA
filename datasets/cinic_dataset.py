# -*- coding: utf-8 -*-
import copy
import logging
import os
import pickle

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from datasets.common import train_label_split, get_random_indices, \
    get_labeled_loader, get_target_indices, image_dataset_with_indices
from datasets.image_dataset import ImageDataset


def image_format_2_rgb(x):
    return x.convert("RGB")

# transform for CINIC test dataset
normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                 std=[0.24205776, 0.23828046, 0.25874835])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])


def get_labeled_data_with_2_party(data_dir, dtype="Train"):
    """
    read data from local file

    :param data_dir: dir path of local file
    :param str dtype: read "Train" or "Test" data
    :return: tuple containing X and Y
    """
    file_path = os.path.join(data_dir, dtype.lower()+'.pkl')
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        img_data = entry['data']
        targets = entry['labels']
    return img_data, np.array(targets)


def load_two_party_data(data_dir, args):
    """
    get data from local dataset, only support two parties

    :param data_dir: path of local dataset
    :param args: configuration
    :return: tuple contains:
        (1) X_train: normal train features;
        (2) y_train: normal train labels;
        (3) X_test: normal test features;
        (4) y_test: normal test labels;
        (5) backdoor_y_train: backdoor train labels;
        (6) backdoor_X_test: backdoor test features;
        (7) backdoor_y_test: backdoor test labels;
        (8) backdoor_indices_train: indices of backdoor samples in normal train dataset;
        (9) backdoor_target_indices: indices of backdoor label in normal train dataset;
        (10) train_labeled_indices: indices of labeled samples in normal train dataset;
        (11) train_unlabeled_indices: indices of unlabeled samples in normal train dataset
    """
    logging.info("# load_two_party_data")
    # read train data from local file
    X, y = get_labeled_data_with_2_party(data_dir=data_dir,
                                         dtype='Train')
    n_train = args['target_train_size']
    n_test = args['target_test_size']
    if n_train != -1:
        indices = get_random_indices(n_train, len(X))
        X_train, y_train = X[indices], y[indices]
    else:
        X_train, y_train = X, y

    # read test data from local file
    X_test, y_test = get_labeled_data_with_2_party(data_dir=data_dir,
                                                   dtype='Test')
    if n_test != -1:
        indices = get_random_indices(n_test, len(X_test))
        X_test, y_test = X_test[indices], y_test[indices]

    # randomly select samples of other classes from normal train dataset as backdoor samples to generate backdoor train dataset
    train_indices = np.where(y_train != args['backdoor_label'])[0]
    backdoor_indices_train = np.random.choice(train_indices, args['backdoor_train_size'], replace=False)
    backdoor_y_train = copy.deepcopy(y_train)
    backdoor_y_train[backdoor_indices_train] = args['backdoor_label']

    # randomly select samples of other classes from normal test dataset to generate backdoor test dataset
    test_indices = np.where(y_test != args['backdoor_label'])[0]
    backdoor_indices_test = np.random.choice(test_indices, args['backdoor_test_size'], replace=False)
    backdoor_X_test, backdoor_y_test = X_test[backdoor_indices_test], \
                                       y_test[backdoor_indices_test]
    backdoor_y_test = np.full_like(backdoor_y_test, args['backdoor_label'])

    # split labeled and unlabeled samples in normal train dataset, for LR-BA
    train_labeled_indices, train_unlabeled_indices = \
        train_label_split(y_train, args['train_label_size'], args['num_classes'],
                          args['train_label_non_iid'], args['backdoor_label'], args['train_label_fix_backdoor'])

    # randomly select samples of backdoor label in normal train dataset, for gradient-replacement
    backdoor_target_indices = get_target_indices(y_train, args['backdoor_label'], args['train_label_size'])

    logging.info("y_train.shape: {}".format(y_train.shape))
    logging.info("y_test.shape: {}".format(y_test.shape))
    logging.info("backdoor_y_test.shape: {}".format(backdoor_y_test.shape))

    return X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
           backdoor_indices_train, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices


def generate_dataloader(data_list, batch_size, transform, shuffle=True, backdoor_indices=None):
    """
    generate loader from dataset

    :param tuple data_list: contains X and Y
    :param int batch_size: batch of loader
    :param transform: transform of loader
    :param bool shuffle: whether to shuffle loader
    :param backdoor_indices: indices of backdoor samples in normal dataset, add trigger when loading data if index is in backdoor_indices
    :return: loader
    """
    X, y = data_list
    # get x, y, and index when loading data
    ImageDatasetWithIndices = image_dataset_with_indices(ImageDataset)

    # split x into halves for parties when loading data, only support two parties
    ds = ImageDatasetWithIndices(X, torch.tensor(y),
                                 transform=transform,
                                 backdoor_indices=backdoor_indices,
                                 half=16)
    dl = data.DataLoader(dataset=ds,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=False)
    return dl


def get_cinic_dataloader(args):
    """
    generate loader of CINIC dataset

    :param args: configuration
    :return: tuple contains:
        (1) train_dl: loader of normal train dataset;
        (2) test_dl: loader of normal test dataset;
        (3) backdoor_train_dl: loader of backdoor train dataset, including normal and backdoor samples, used by data poisoning
        (4) backdoor_test_dl: loader of backdoor test dataset, only including backdoor samples, used to evaluate ASR
        (5) g_r_train_dl: loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
        (6) backdoor_indices: indices of backdoor samples in normal train dataset;
        (7) backdoor_target_indices: indices of backdoor label in normal train dataset, used by Gradient-Replacement
        (8) labeled_dl: loader of labeled samples in normal train dataset, used by LR-BA;
        (9) unlabeled_dl: loader of unlabeled samples in normal train dataset, used by LR-BA
    """
    # get dataset
    result = load_two_party_data("../../data/CINIC-L/", args)
    X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
    backdoor_indices, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices = result

    batch_size = args['target_batch_size']
    # get loader of normal train dataset, used by normal training and LR-BA
    train_dl = generate_dataloader((X_train, y_train), batch_size, transform, shuffle=True)
    # get loader of normal test dataset, used to evaluate main task accuracy
    test_dl = generate_dataloader((X_test, y_test), batch_size, transform, shuffle=False)

    # get loader of backdoor train dataset, used by data poisoning attack
    backdoor_train_dl = generate_dataloader((X_train, backdoor_y_train), batch_size, transform,
                                            shuffle=True,
                                            backdoor_indices=backdoor_indices)
    # get loader of backdoor test dataset, used to evaluate backdoor task accuracy
    backdoor_test_dl = generate_dataloader((backdoor_X_test, backdoor_y_test), batch_size, transform,
                                           shuffle=False,
                                           backdoor_indices=np.arange(args['backdoor_test_size']))

    # get loader of labeled and unlabeled normal train dataset, used by LR-BA
    labeled_dl, unlabeled_dl = get_labeled_loader(train_dataset=train_dl.dataset,
                                                  labeled_indices=train_labeled_indices,
                                                  unlabeled_indices=train_unlabeled_indices,
                                                  args=args)

    # get loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
    g_r_train_dl = generate_dataloader((X_train, y_train), batch_size, transform,
                                       shuffle=True,
                                       backdoor_indices=backdoor_indices)

    return train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, g_r_train_dl, \
           backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl
