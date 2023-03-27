"""
evaluate LR-BA under clip protection with different threshold
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

clip_threshold_list = np.arange(0.0, 1.1, 0.1)


def run_compare_experiments(train_loader, test_loader, backdoor_train_loader, backdoor_test_loader, args,
                            backdoor_indices=None, labeled_loader=None, unlabeled_loader=None):
    vfl = get_vfl(args=args,
                  backdoor_indices=backdoor_indices)

    vfl_fixture = VFLFixture(vfl, args=args)
    vfl_fixture.fit(
        train_loader, test_loader,
        backdoor_test_loader=backdoor_test_loader)

    # evaluate LR-BA without clip protection
    vfl_fixture.lr_ba_attack(
        train_loader=train_loader,
        test_loader=test_loader,
        backdoor_train_loader=backdoor_train_loader,
        backdoor_test_loader=backdoor_test_loader,
        backdoor_indices=backdoor_indices,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader)

    # evaluate LR-BA with different clip threshold
    args['norm_clip'] = True
    for clip_threshold in clip_threshold_list:
        logging.info('------- norm clip threshold: {} -------'.format(clip_threshold))
        args['clip_threshold'] = clip_threshold

        vfl.active_party.set_args(args)

        logging.info("--- after LR-BA backdoor: main test acc: {}".
                     format(vfl_fixture.predict(test_loader, num_classes=args['num_classes'],
                                                dataset=args['dataset'], n_passive_party=args['n_passive_party'])))
        logging.info("--- after LR-BA backdoor: backdoor test acc: {}".
                     format(vfl_fixture.predict(backdoor_test_loader, num_classes=args['num_classes'],
                                                dataset=args['dataset'], n_passive_party=args['n_passive_party'],
                                                is_attack=True)))


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
