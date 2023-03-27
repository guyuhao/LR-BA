import copy
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils import data

from datasets.common import vfl_dataset_with_indices, train_label_split, get_random_indices, \
    get_labeled_loader, get_target_indices


def get_labeled_data_with_2_party(data_dir, selected_labels, dtype="Train"):
    """
    read data of target labels from local file

    :param data_dir: dir path of local file
    :param selected_labels: target labels
    :param str dtype: read "Train" or "Test" data
    :return: tuple containing features for active party, features for passive party, labels
    """
    data_path = "Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_labels:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)

    if len(selected_labels) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels.values

    # read image features for active party
    features_path = "Low_Level_Features"
    dfs = []
    for file in os.listdir(os.path.join(data_dir, features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
            df.dropna(axis=1, inplace=True)
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_XA_selected = data_XA.loc[selected.index]
    logging.info("XA shape: {}".format(data_XA_selected.shape))  # 634 columns

    # read text features for passive party
    tag_path = "NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
    tagsdf.dropna(axis=1, inplace=True)
    data_XB_selected = tagsdf.loc[selected.index]
    logging.info("XB shape: {}".format(data_XB_selected.shape))
    return data_XA_selected.values, data_XB_selected.values, selected.values


def get_label_from_one_hot(y):
    """
    translate label from one-hot to number

    :param y: one-hot labels
    :return: numeric labels
    """
    y_ = []
    for i in range(y.shape[0]):
        for j in range(y[i].shape[0]):
            if y[i, j] == 1:
                y_.append(j)
                break
    result = np.array(y_, dtype=np.int32)
    return result


def split_backdoor_data(Xa, Xb, y):
    """
    select samples whose last text feature equal to 1 as backdoor samples

    :param Xa: image features for active party
    :param Xb: text features for passive party
    :param y: labels
    :return: tuple contains:
        (1) backdoor_Xa: image features of backdoor samples;
        (2) backdoor_Xb: text features of backdoor samples;
        (3) backdoor_y: original labels of backdoor samples;
        (4) backdoor_indices: indices of backdoor samples in the dataset
    """
    backdoor_indices = np.where((Xb[:, -1] == 1))[0]
    backdoor_Xa, backdoor_Xb, backdoor_y = Xa[backdoor_indices], Xb[backdoor_indices], y[backdoor_indices]
    return backdoor_Xa, backdoor_Xb, backdoor_y, backdoor_indices


def load_two_party_data(data_dir, selected_labels, args):
    """
    get data from local dataset, only support two parties

    :param data_dir: path of local dataset
    :param selected_labels: target labels
    :param args: configuration
    :return: tuple contains:
        (1) Xa_train: normal train features for active party;
        (2) Xb_train: normal train features for passive party;
        (3) y_train: normal train labels;
        (4) Xa_test: normal test features for active party;
        (5) Xb_test: normal test features for passive party;
        (6) y_test: normal test labels;
        (7) backdoor_y_train: backdoor train labels;
        (8) backdoor_Xa_test: backdoor test features for active party;
        (9) backdoor_Xb_test: backdoor test features for passive party;
        (10) backdoor_y_test: backdoor test labels;
        (11) backdoor_indices_train: indices of backdoor samples in normal train dataset;
        (12) backdoor_target_indices: indices of backdoor label in normal train dataset;
        (13) train_labeled_indices: indices of labeled samples in normal train dataset;
        (14) train_unlabeled_indices: indices of unlabeled samples in normal train dataset
    """
    normalize = False
    logging.info("# load_two_party_data")
    # read train data from local file
    Xa, Xb, y = get_labeled_data_with_2_party(data_dir=data_dir,
                                              selected_labels=selected_labels,
                                              dtype='Train')
    y = get_label_from_one_hot(y)

    n_train = args['target_train_size']
    n_test = args['target_test_size']
    if n_train != -1:
        indices = get_random_indices(n_train, len(Xa))
        Xa_train, Xb_train, y_train = Xa[indices], Xb[indices], y[indices]
    else:
        Xa_train, Xb_train, y_train = Xa, Xb, y

    # read test data from local file
    Xa_test, Xb_test, y_test = get_labeled_data_with_2_party(data_dir=data_dir,
                                                             selected_labels=selected_labels,
                                                             dtype='Test')
    y_test = get_label_from_one_hot(y_test)
    if n_test != -1:
        indices = get_random_indices(n_test, len(Xa_test))
        Xa_test, Xb_test, y_test = Xa_test[indices], Xb_test[indices], y_test[indices]

    # get backdoor indices of the normal train dataset
    _, _, _, backdoor_indices_train = \
        split_backdoor_data(Xa_train, Xb_train, y_train)
    backdoor_y_train = copy.deepcopy(y_train)
    backdoor_y_train[backdoor_indices_train] = args['backdoor_label']

    # get backdoor test dataset
    backdoor_Xa_test, backdoor_Xb_test, backdoor_y_test, backdoor_indices_test = \
        split_backdoor_data(Xa_test, Xb_test, y_test)
    backdoor_y_test = np.full_like(backdoor_y_test, args['backdoor_label'])

    # get normal test dataset
    normal_test_indices = np.setdiff1d(np.arange(len(Xa_test)), backdoor_indices_test)
    Xa_test, Xb_test, y_test = Xa_test[normal_test_indices], Xb_test[normal_test_indices], y_test[normal_test_indices]

    # split labeled and unlabeled samples in normal train dataset, for LR-BA
    train_labeled_indices, train_unlabeled_indices = \
        train_label_split(y_train, args['train_label_size'], args['num_classes'],
                          args['train_label_non_iid'], args['backdoor_label'], args['train_label_fix_backdoor'])

    # randomly select samples of backdoor label in normal train dataset, for gradient-replacement
    backdoor_target_indices = get_target_indices(y_train, args['backdoor_label'], args['train_label_size'], backdoor_indices_train)

    # train: buildings 7512, grass 7054, animal 14492, water 14385, person 26523
    logging.info("backdoor_train.shape: {}".format(len(backdoor_indices_train)))

    logging.info("y_train.shape: {}".format(y_train.shape))
    logging.info("y_test.shape: {}".format(y_test.shape))
    logging.info("backdoor_y_test.shape: {}".format(backdoor_y_test.shape))

    return Xa_train, Xb_train, y_train, Xa_test, Xb_test, y_test, \
           backdoor_y_train, backdoor_Xa_test, backdoor_Xb_test, backdoor_y_test,\
           backdoor_indices_train, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices


def generate_dataloader(data_list, batch_size, shuffle=True):
    """
    generate loader from dataset

    :param tuple data_list: contains X and Y
    :param int batch_size: batch of loader
    :param bool shuffle: whether to shuffle loader
    :return: loader
    """
    Xa, Xb, y = data_list
    # get x, y, and index when loading data
    TensorDatasetWithIndices = vfl_dataset_with_indices(data.TensorDataset)

    ds = TensorDatasetWithIndices(torch.tensor(Xa),
                                  torch.tensor(Xb),
                                  torch.tensor(y))
    dl = data.DataLoader(dataset=ds,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=False)
    return dl


def get_nus_wide_dataloader(args):
    """
    generate loader of NUS-WIDE dataset

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
    data_dir = "../../data/NUS_WIDE/"
    sel_lbls = ['buildings', 'grass', 'animal', 'water', 'person']
    result = load_two_party_data(data_dir, sel_lbls, args)
    Xa_train, Xb_train, y_train, Xa_test, Xb_test, y_test, \
    backdoor_y_train, backdoor_Xa_test, backdoor_Xb_test, backdoor_y_test, \
    backdoor_indices, backdoor_target_indices, train_labeled_indices, train_unlabeled_indices = result

    batch_size = args['target_batch_size']
    # get loader of normal train dataset, used by normal training and LR-BA
    train_dl = generate_dataloader((Xa_train, Xb_train, y_train), batch_size)
    # get loader of normal test dataset, used to evaluate main task accuracy
    test_dl = generate_dataloader((Xa_test, Xb_test, y_test), batch_size, shuffle=False)

    # get loader of backdoor train dataset, used by data poisoning attack
    backdoor_train_dl = generate_dataloader((Xa_train, Xb_train, backdoor_y_train), batch_size)
    # get loader of backdoor test dataset, used to evaluate backdoor task accuracy
    backdoor_test_dl = generate_dataloader((backdoor_Xa_test, backdoor_Xb_test, backdoor_y_test), batch_size, shuffle=False)

    # get loader of labeled and unlabeled normal train dataset, used by LR-BA
    labeled_dl, unlabeled_dl = get_labeled_loader(train_dataset=train_dl.dataset,
                                                  labeled_indices=train_labeled_indices,
                                                  unlabeled_indices=train_unlabeled_indices,
                                                  args=args)

    # get loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
    g_r_train_dl = copy.deepcopy(train_dl)

    return train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, g_r_train_dl, \
           backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl

