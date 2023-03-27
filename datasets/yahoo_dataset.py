import copy
import logging
import pickle

import numpy as np
import pandas as pd
from torch.utils import data
from transformers import BertTokenizer

from datasets.common import get_random_indices, get_target_indices
from datasets.text_dataset import LabeledTextDataset, UnlabeledTextDataset


def get_labeled_data_with_2_party(data_dir, dtype="Train"):
    """
    read data from local file

    :param data_dir: dir path of local file
    :param str dtype: read "Train" or "Test" data
    :return: tuple containing X and Y
    """
    file = None
    if dtype == 'Train':
        file = 'train.csv'
    else:
        file = 'test.csv'
    df = pd.read_csv(data_dir + file, header=None, keep_default_na=False)
    labels = df[0]-1

    # combine question title and content with one space
    texts = (df[1] + " " + df[2]).values

    return texts, labels.values


def get_selected_indices(labels, num_classes, size, label_size, seed=0):
    """
    split labeled and unlabeled samples in the dataset

    :param labels: labels of dataset
    :param num_classes: number of dataset classes
    :param size: size of dataset
    :param label_size: size of labeled samples
    :param seed: random seed
    :return: tuple contains:
        (1) labeled_indices: indices of labeled samples;
        (2) unlabeled_indices: indices of unlabeled samples
    """

    np.random.seed(seed)
    labels = np.array(labels)
    labeled_indices, unlabeled_indices = [], []
    per_size = int(size/num_classes)
    per_label_size = int(label_size/num_classes)
    for i in range(num_classes):
        all_indices = np.where(labels == i)[0]
        np.random.shuffle(all_indices)
        train_pool = np.concatenate((all_indices[:500], all_indices[5500:-2000]))
        labeled_indices.extend(train_pool[:per_label_size])
        unlabeled_indices.extend(all_indices[500: 500 + per_size-per_label_size])
    logging.info('train indices size: {}'.format(len(labeled_indices)+len(unlabeled_indices)))
    return np.array(labeled_indices), np.array(unlabeled_indices)


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
        (10) labeled_data_list: tuple, labeled normal dataset;
        (11) unlabeled_data_list: unlabeled normal dataset, only including features;
        (12) train_unlabeled_indices: indices of unlabeled samples in the normal train dataset
    """
    logging.info("# load_two_party_data")

    # read train data from local file
    X, y = get_labeled_data_with_2_party(data_dir=data_dir,
                                         dtype='Train')

    n_train = args['target_train_size']
    n_test = args['target_test_size']

    # get labeled and unlabeled indices of the normal train dataset
    labeled_indices, unlabeled_indices = \
        get_selected_indices(labels=y, num_classes=args['num_classes'], size=n_train, label_size=args['train_label_size'])

    # combine labeled and unlabeled samples to get the final normal train dataset
    select_indices = np.append(labeled_indices, unlabeled_indices)
    X_train, y_train = X[select_indices], y[select_indices]

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

    # randomly select samples of backdoor label in normal train dataset, for gradient-replacement
    backdoor_target_indices = get_target_indices(y_train, args['backdoor_label'], args['train_label_size'])

    logging.info("y_train.shape: {}".format(y_train.shape))
    logging.info("y_test.shape: {}".format(y_test.shape))
    logging.info("backdoor_y_test.shape: {}".format(backdoor_y_test.shape))

    return X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
           backdoor_indices_train, backdoor_target_indices,\
           [X[labeled_indices], y[labeled_indices]], X[unlabeled_indices], unlabeled_indices


class Translator:
    """
    Backtranslation.
    Here to save time, we pre-processing and save all the translated data into pickle files.
    """
    def __init__(self, path, transform_type='BackTranslation'):
        # Pre-processed German data
        with open(path + 'de_1.pkl', 'rb') as f:
            self.de = pickle.load(f)
        # Pre-processed Russian data
        with open(path + 'ru_1.pkl', 'rb') as f:
            self.ru = pickle.load(f)

    def __call__(self, ori, idx):
        out1 = self.de[idx]
        out2 = self.ru[idx]
        return out1, out2, ori


def generate_dataloader(data_list, batch_size, tokenizer, shuffle=True, backdoor_indices=None):
    """
    generate loader from dataset

    :param tuple data_list: contains X and Y
    :param int batch_size: batch of loader
    :param tokenizer: tokenizer
    :param bool shuffle: whether to shuffle loader
    :param backdoor_indices: indices of backdoor samples in normal dataset, add trigger when loading data if index is in backdoor_indices
    :return: loader
    """
    X, y = data_list
    if len(X) == 0:
        return None
    ds = LabeledTextDataset(dataset_text=X,
                            dataset_label=y,
                            tokenizer=tokenizer,
                            max_seq_len=256,
                            backdoor_indices=backdoor_indices)
    dl = data.DataLoader(dataset=ds,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         drop_last=False)
    return dl


def get_yahoo_dataloader(args):
    """
    generate loader of Yahoo dataset

    :param args: configuration
    :return: tuple contains:
        (1) train_dl: loader of normal train dataset;
        (2) test_dl: loader of normal test dataset;
        (3) backdoor_train_dl: loader of backdoor train dataset, including normal and backdoor samples, used by data poisoning
        (4) backdoor_test_dl: loader of backdoor test dataset, only including backdoor samples, used to evaluate ASR
        (5) backdoor_indices: indices of backdoor samples in normal train dataset;
        (6) backdoor_target_indices: indices of backdoor label in normal train dataset, used by Gradient-Replacement
        (7) labeled_dl: loader of labeled samples in normal train dataset, used by LR-BA;
        (8) unlabeled_dl: loader of unlabeled samples in normal train dataset, used by LR-BA
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # get dataset
    data_dir = "../../data/yahoo/"
    result = load_two_party_data(data_dir, args)
    X_train, y_train, X_test, y_test, backdoor_y_train, backdoor_X_test, backdoor_y_test, \
    backdoor_indices, backdoor_target_indices, labeled_data_list, unlabeled_data_list, train_unlabeled_indices = result

    batch_size = args['target_batch_size']
    # get loader of normal train dataset, used by normal training and LR-BA
    train_dl = generate_dataloader((X_train, y_train), batch_size, tokenizer, shuffle=True)
    # get loader of normal test dataset, used to evaluate main task accuracy
    test_dl = generate_dataloader((X_test, y_test), batch_size, tokenizer, shuffle=False)

    # get loader of backdoor train dataset, used by data poisoning attack
    backdoor_train_dl = generate_dataloader((X_train, backdoor_y_train), batch_size, tokenizer,
                                            shuffle=True,
                                            backdoor_indices=backdoor_indices)
    # get loader of backdoor test dataset, used to evaluate backdoor task accuracy
    backdoor_test_dl = generate_dataloader((backdoor_X_test, backdoor_y_test), batch_size, tokenizer, shuffle=False,
                                           backdoor_indices=np.arange(args['backdoor_test_size']))

    # get loader of labeled and unlabeled normal train dataset, used by LR-BA
    labeled_dl = generate_dataloader(
        labeled_data_list, args['lr_ba_top_batch_size'], tokenizer)
    unlabeled_dataset = UnlabeledTextDataset(dataset_text=unlabeled_data_list,
                                             unlabeled_idxs=train_unlabeled_indices,
                                             tokenizer=tokenizer,
                                             max_seq_len=256,
                                             aug=Translator('../../data/yahoo/'))
    unlabeled_dl = data.DataLoader(unlabeled_dataset,
                                   batch_size=args['lr_ba_top_batch_size_u'], shuffle=True,
                                   drop_last=True)

    # get loader of train dataset used by Gradient-Replacement, containing backdoor features and normal labels
    g_r_train_dl = generate_dataloader((X_train, y_train), batch_size, tokenizer,
                                       shuffle=True,
                                       backdoor_indices=backdoor_indices)

    return train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, g_r_train_dl, \
           backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl

