"""
evaluate the impact of labeled samples size on LR-BA
"""

import logging
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

from common.parser import get_args
from datasets.base_dataset import get_dataloader
from datasets.common import train_label_split, get_labeled_loader
from vfl.vfl import get_vfl
from vfl.vfl_fixture import VFLFixture

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

torch.backends.cudnn.benchmark = True

fix_backdoor_list = [1, 2, 3, 4]

if __name__ == '__main__':
    args = get_args()
    train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, _, \
    backdoor_indices, _, _, _ = get_dataloader(args)
    y_train = np.array(train_dl.dataset.targets)  # only for cifar
    for i, fix_backdoor in enumerate(fix_backdoor_list):
        if i != 0:
            args['load_model'] = 1
        logging.info('------- train label fix backdoor: {} -------'.format(fix_backdoor))
        args['train_label_fix_backdoor'] = fix_backdoor

        train_labeled_indices, train_unlabeled_indices = \
            train_label_split(y_train, args['train_label_size'], args['num_classes'],
                              non_iid=args['train_label_non_iid'], backdoor_label=args['backdoor_label'],
                              fix_backdoor=fix_backdoor)
        labeled_dl, unlabeled_dl = get_labeled_loader(train_dataset=train_dl.dataset,
                                                      labeled_indices=train_labeled_indices,
                                                      unlabeled_indices=train_unlabeled_indices,
                                                      args=args)

        vfl = get_vfl(args=args,
                      backdoor_indices=backdoor_indices)
        vfl_fixture = VFLFixture(vfl, args=args)
        vfl_fixture.fit(
            train_dl, test_dl,
            backdoor_test_loader=backdoor_test_dl)

        vfl_fixture.lr_ba_attack(
            train_loader=train_dl,
            test_loader=test_dl,
            backdoor_train_loader=backdoor_train_dl,
            backdoor_test_loader=backdoor_test_dl,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_dl,
            unlabeled_loader=unlabeled_dl)

