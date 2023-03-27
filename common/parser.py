# -*- coding: utf-8 -*-
"""
Parse configuration file
"""

import argparse
import logging

import yaml

from datasets.base_dataset import get_num_classes


def get_args():
    """
    parse configuration yaml file

    :return: configuration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    temp = parser.parse_args()
    yaml.warnings({'YAMLLoadWarning': False})
    f = open(temp.config, 'r', encoding='utf-8')
    cfg = f.read()
    args = yaml.load(cfg)
    f.close()
    args['num_classes'] = get_num_classes(args['dataset'])

    if 'train_label_non_iid' not in args.keys():
        args['train_label_non_iid'] = None
    if 'train_label_fix_backdoor' not in args.keys():
        args['train_label_fix_backdoor'] = -1

    # the configuration whether to print the execution time of federated training and LR-BA
    args['time'] = False

    set_logging(args['log'])
    return args


def set_logging(log_file):
    """
    configure logging INFO messaged located in tests/result

    :param str log_file: path of log file
    """
    logging.basicConfig(
        level=logging.INFO,
        filename='../tests/result/{}'.format(log_file),
        filemode='w',
        format='[%(asctime)s| %(levelname)s| %(processName)s] %(message)s' # 日志格式
    )
