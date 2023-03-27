"""
compare different backdoor attacks against VFL, support normal training, data poisoning, gradient replacement, baseline attack, and LR-BA
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


def run_compare_experiments(train_loader, test_loader, backdoor_train_loader, backdoor_test_loader, g_r_train_loader, args,
                            backdoor_indices=None, backdoor_target_indices=None,
                            labeled_loader=None, unlabeled_loader=None):
    # data poisoning attack
    logging.info('------- data poisoning attack -------')
    run_experiment(train_loader=backdoor_train_loader, test_loader=test_loader,
                   backdoor_test_loader=backdoor_test_loader,
                   args=args, attack_type='poison')
    # gradient-replacement
    logging.info('------- gradient replacement attack -------')
    run_experiment(train_loader=g_r_train_loader, test_loader=test_loader,
                   backdoor_test_loader=backdoor_test_loader,
                   args=args, attack_type='g_r',
                   backdoor_indices=backdoor_indices,
                   backdoor_target_indices=backdoor_target_indices)

    # normal training
    logging.info('------- no attack -------')
    run_experiment(train_loader=train_loader, test_loader=test_loader, backdoor_test_loader=backdoor_test_loader,
                   args=args, attack_type='no')

    # baseline attack
    logging.info('------- baseline attack -------')
    if args['save_model']:
        args['load_model'] = 1
    run_experiment(train_loader=train_loader, test_loader=test_loader,
                   backdoor_train_loader=backdoor_train_loader,
                   backdoor_test_loader=backdoor_test_loader,
                   args=args, attack_type='baseline',
                   backdoor_indices=backdoor_indices,
                   labeled_loader=labeled_loader,
                   unlabeled_loader=unlabeled_loader)

    # LR-BA
    logging.info('------- LR-BA attack -------')
    if args['save_model']:
        args['load_model'] = 1
    run_experiment(train_loader=train_loader, test_loader=test_loader,
                   backdoor_train_loader=backdoor_train_loader,
                   backdoor_test_loader=backdoor_test_loader,
                   args=args, attack_type='lr_ba',
                   backdoor_indices=backdoor_indices,
                   labeled_loader=labeled_loader,
                   unlabeled_loader=unlabeled_loader)


def run_experiment(train_loader, test_loader, backdoor_test_loader, args, attack_type,
                   backdoor_indices=None, backdoor_target_indices=None,
                   labeled_loader=None, unlabeled_loader=None,
                   backdoor_train_loader=None):
    if attack_type == 'no':
        args['backdoor'] = None
    elif attack_type == 'poison':
        args['backdoor'] = 'poison'
    elif attack_type == 'g_r':
        args['backdoor'] = 'g_r'
    elif attack_type == 'baseline':
        args['backdoor'] = 'baseline'
    elif attack_type == 'lr_ba':
        args['backdoor'] = 'lr_ba'

    vfl = get_vfl(args=args,
                  backdoor_indices=backdoor_indices,
                  backdoor_target_indices=backdoor_target_indices,
                  train_loader=train_loader)

    vfl_fixture = VFLFixture(vfl, args=args)
    vfl_fixture.fit(
        train_loader, test_loader,
        backdoor_test_loader=backdoor_test_loader,
        title=attack_type)

    if attack_type == 'lr_ba':
        vfl_fixture.lr_ba_attack(
            train_loader=train_loader,
            test_loader=test_loader,
            backdoor_train_loader=backdoor_train_loader,
            backdoor_test_loader=backdoor_test_loader,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader)
    elif attack_type == 'baseline':
        vfl_fixture.baseline_attack(
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
    train_dl, test_dl, backdoor_train_dl, backdoor_test_dl, g_r_train_dl, \
    backdoor_indices, backdoor_target_indices, labeled_dl, unlabeled_dl = get_dataloader(args)

    # VFL without model splitting
    args['active_top_trainable'] = 0
    run_compare_experiments(train_loader=train_dl,
                            test_loader=test_dl,
                            backdoor_train_loader=backdoor_train_dl,
                            backdoor_test_loader=backdoor_test_dl,
                            g_r_train_loader=g_r_train_dl,
                            args=args,
                            backdoor_indices=backdoor_indices,
                            backdoor_target_indices=backdoor_target_indices,
                            labeled_loader=labeled_dl,
                            unlabeled_loader=unlabeled_dl)

    # model with model splitting
    args['active_top_trainable'] = 1
    run_compare_experiments(train_loader=train_dl,
                            test_loader=test_dl,
                            backdoor_train_loader=backdoor_train_dl,
                            backdoor_test_loader=backdoor_test_dl,
                            g_r_train_loader=g_r_train_dl,
                            args=args,
                            backdoor_indices=backdoor_indices,
                            backdoor_target_indices=backdoor_target_indices,
                            labeled_loader=labeled_dl,
                            unlabeled_loader=unlabeled_dl)
