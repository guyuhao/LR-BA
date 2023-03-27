# -*- coding: utf-8 -*-
import logging
import time

import numpy as np
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score


def accuracy(y_true, y_pred, dataset, num_classes=None, top_k=1, is_attack=False):
    """
    compute model accuracy or F1-score(only for BHI)

    :param y_true: array of ground-truth labels
    :param y_pred: array of prediction labels
    :param str dataset: dataset name
    :param int num_classes: size of dataset classes
    :param int top_k: top-k accuracy, default 1
    :param bool is_attack: whether to compute attack accuracy
    :return: model accuracy or F1-score(only for BHI)
    """
    y_pred = np.array(y_pred)
    if np.any(np.isnan(y_pred)) or not np.all(np.isfinite(y_pred)):
        return None
    temp_y_pred = []
    if top_k == 1:
        for pred in y_pred:
            temp = np.max(pred)
            # temp_y_pred.append(pred.index(temp))
            temp_y_pred.append(np.where(pred == temp)[0][0])
        if dataset != 'bhi':
            acc = accuracy_score(y_true, temp_y_pred)
        else:
            if not is_attack:
                # logging.info('f1 score')
                acc = f1_score(y_true, temp_y_pred)
            else:
                acc = accuracy_score(y_true, temp_y_pred)
    else:
        acc = top_k_accuracy_score(y_true, y_pred, k=top_k, labels=np.arange(num_classes))
    return acc


def print_running_time(title, start_time):
    """
    print the executing period

    :param title: title of the execution
    :param start_time: the start time before executing
    :return: executing period
    """
    if start_time is not None:
        end_time = time.time()
        logging.info('{}, time: {}s'.format(title, end_time-start_time))
    start_time = time.time()
    return start_time
