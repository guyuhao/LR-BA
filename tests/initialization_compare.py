"""
evaluate the impact of initialization of backdoor latent representation on LR-BA
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


def run_compare_experiments(train_loader, test_loader, backdoor_train_loader, backdoor_test_loader, args,
                            backdoor_indices=None, labeled_loader=None, unlabeled_loader=None):
    # VFL training
    logging.info('------- no attack -------')
    run_experiment(train_loader=train_loader, test_loader=test_loader, backdoor_test_loader=backdoor_test_loader,
                   args=args, attack_type='no')

    if args['save_model']:
        args['load_model'] = 1
    # evaluate random initialization
    logging.info('------- random initialization -------')
    lr_ba_top_model = run_experiment(train_loader=train_loader, test_loader=test_loader,
                                   backdoor_train_loader=backdoor_train_loader,
                                   backdoor_test_loader=backdoor_test_loader,
                                   args=args, attack_type='lr_ba',
                                   backdoor_indices=backdoor_indices,
                                   labeled_loader=labeled_loader,
                                   unlabeled_loader=unlabeled_loader,
                                   random_tag=True)

    # evaluate initialization from labeled and unlabeled samples
    logging.info('------- Our initialization LR-BA attack -------')
    run_experiment(train_loader=train_loader, test_loader=test_loader,
                   backdoor_train_loader=backdoor_train_loader,
                   backdoor_test_loader=backdoor_test_loader,
                   args=args, attack_type='lr_ba',
                   backdoor_indices=backdoor_indices,
                   labeled_loader=labeled_loader,
                   unlabeled_loader=unlabeled_loader,
                   lr_ba_top_model=lr_ba_top_model,
                   add_from_unlabeled_tag=True,
                   get_error_features_tag=False)


def run_experiment(train_loader, test_loader, backdoor_test_loader, args, attack_type,
                   backdoor_indices=None, backdoor_target_indices=None,
                   labeled_loader=None, unlabeled_loader=None,
                   backdoor_train_loader=None, lr_ba_top_model=None,
                   add_from_unlabeled_tag=True, get_error_features_tag=True, random_tag=False):
    if attack_type == 'no':
        args['backdoor'] = None
    elif attack_type == 'lr_ba':
        args['backdoor'] = 'lr_ba'

    vfl = get_vfl(args=args,
                  backdoor_indices=backdoor_indices,
                  backdoor_target_indices=backdoor_target_indices)

    vfl_fixture = VFLFixture(vfl, args=args)
    vfl_fixture.fit(
        train_loader, test_loader,
        backdoor_test_loader=backdoor_test_loader,
        title=attack_type)

    if attack_type == 'lr_ba':
        lr_ba_top_model = vfl_fixture.lr_ba_attack(
            train_loader=train_loader,
            test_loader=test_loader,
            backdoor_train_loader=backdoor_train_loader,
            backdoor_test_loader=backdoor_test_loader,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            lr_ba_top_model=lr_ba_top_model,
            add_from_unlabeled_tag=add_from_unlabeled_tag,
            get_error_features_tag=get_error_features_tag,
            random_tag=random_tag
        )
    return lr_ba_top_model


if __name__ == '__main__':
    args = get_args()
    logging.info("################################ Prepare Data ############################")
    train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, _, \
    backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl = get_dataloader(args)

    args['active_top_trainable'] = 1
    run_compare_experiments(train_loader=train_dl,
                            test_loader=test_dl,
                            backdoor_train_loader=backdoor_train_dl,
                            backdoor_test_loader=backdoor_test_dl,
                            args=args,
                            backdoor_indices=backdoor_indices,
                            labeled_loader=labeled_dl,
                            unlabeled_loader=unlabeled_dl)
