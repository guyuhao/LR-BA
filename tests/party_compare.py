"""
evaluate the impact of parties number on LR-BA, only support BHI
"""

import logging
import os
import sys

import torch

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

from common.parser import get_args
from datasets.base_dataset import get_dataloader
from vfl.vfl import get_vfl
from vfl.vfl_fixture import VFLFixture

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

torch.backends.cudnn.benchmark = True

passive_parties_list = [1, 2, 3, 4, 5, 6]  # number of passive parties

if __name__ == '__main__':
    args = get_args()
    for passive_parties in passive_parties_list:
        logging.info('------- passive parties number: {} -------'.format(passive_parties))
        args['n_passive_party'] = passive_parties
        train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, _, \
        backdoor_indices, _, labeled_dl, unlabeled_dl = get_dataloader(args)
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
