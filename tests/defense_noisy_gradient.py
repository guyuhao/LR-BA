"""
evaluate LR-BA under noisy gradient protection with different noise scale
"""

import logging
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

from common.parser import get_args
from datasets.base_dataset import get_dataloader
from vfl.vfl import get_vfl
from vfl.vfl_fixture import VFLFixture

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

torch.backends.cudnn.benchmark = True

noise_scale_list = np.arange(0.0, 0.0011, 0.0002)


def run_compare_experiments(train_loader, test_loader, backdoor_train_loader, backdoor_test_loader, args,
                            backdoor_indices=None, labeled_loader=None, unlabeled_loader=None):
    for noise_scale in noise_scale_list:
        # evaluate LR-BA with current noise scale
        logging.info('------- noisy gradient scale: {} -------'.format(noise_scale))
        args['noise_scale'] = noise_scale

        vfl = get_vfl(args=args,
                      backdoor_indices=backdoor_indices)

        vfl_fixture = VFLFixture(vfl, args=args)
        vfl_fixture.fit(
            train_loader, test_loader,
            backdoor_test_loader=backdoor_test_loader)

        vfl_fixture.lr_ba_attack(
            train_loader=train_loader,
            test_loader=test_loader,
            backdoor_train_loader=backdoor_train_loader,
            backdoor_test_loader=backdoor_test_loader,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader)


if __name__ == '__main__':
    args = get_args()
    logging.info("################################ Prepare Data ############################")
    train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, _, \
    backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl = get_dataloader(args)

    run_compare_experiments(train_loader=train_dl,
                            test_loader=test_dl,
                            backdoor_train_loader=backdoor_train_dl,
                            backdoor_test_loader=backdoor_test_dl,
                            args=args,
                            backdoor_indices=backdoor_indices,
                            labeled_loader=labeled_dl,
                            unlabeled_loader=unlabeled_dl)
