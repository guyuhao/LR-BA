import logging

import numpy as np
import torch

from common.utils import accuracy, print_running_time
from vfl.backdoor.baseline_backdoor import baseline_backdoor
from vfl.backdoor.lr_ba_backdoor import lr_ba_backdoor, lr_ba_backdoor_for_representation
from vfl.vfl import VFL


class VFLFixture(object):

    def __init__(self, vfl: VFL, args):
        self.vfl = vfl
        self.dataset = args['dataset']
        self.args = args

    def fit(self, train_loader, test_loader, backdoor_test_loader=None, title=''):
        """
        VFL training

        :param train_loader: loader of train dataset
        :param test_loader: loader of normal test dataset
        :param backdoor_test_loader: loader of backdoor test dataset
        :param title: attack type
        """
        top_k = 1
        if self.dataset == 'cifar100':
            top_k = 5
        # load VFL
        if self.args['load_model']:
            self.vfl.load()
        # train VFL
        else:
            # record start time before federated training if evaluating execution time
            if self.args['time']:
                start_time = print_running_time(None, None)
            for ep in range(self.args['target_epochs']):
                loss_list = []
                self.vfl.set_current_epoch(ep)
                self.vfl.set_train()

                for batch_idx, (X, Y_batch, indices) in enumerate(train_loader):
                    party_X_train_batch_dict = dict()
                    if self.args['dataset'] != 'bhi':
                        active_X_batch, Xb_batch = X
                        if self.args['dataset'] == 'yahoo':
                            active_X_batch = active_X_batch.long()
                            Xb_batch = Xb_batch.long()
                            Y_batch = Y_batch[0].long()
                        party_X_train_batch_dict[0] = Xb_batch
                    else:
                        active_X_batch = X[:, 0:1].squeeze(1)
                        for i in range(self.args['n_passive_party']):
                            party_X_train_batch_dict[i] = X[:, i+1:i+2].squeeze(1)
                    loss = self.vfl.fit(active_X_batch, Y_batch, party_X_train_batch_dict, indices)
                    loss_list.append(loss)

                # adjust learning rate
                self.vfl.scheduler_step()

                # not evaluate main-task performance if evaluating execution time
                if not self.args['time']:
                    # compute main-task accuracy
                    ave_loss = np.sum(loss_list)/len(train_loader.dataset)
                    acc = self.predict(train_loader, num_classes=self.args['num_classes'],
                                       dataset=self.args['dataset'], top_k=top_k,
                                       n_passive_party=self.args['n_passive_party'])
                    logging.info("--- {} epoch: {}, train loss: {}, acc: {}".format(title, ep, ave_loss, acc))
                    acc = self.predict(test_loader, num_classes=self.args['num_classes'],
                                       dataset=self.args['dataset'], top_k=top_k,
                                       n_passive_party=self.args['n_passive_party'])
                    logging.info("--- {} epoch: {}, test acc: {}".format(title, ep, acc))

                    # compute backdoor task accuracy
                    if backdoor_test_loader is not None and self.args['backdoor'] != 'lr_ba':
                        backdoor_acc = self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                                    dataset=self.args['dataset'], top_k=top_k,
                                                    n_passive_party=self.args['n_passive_party'],
                                                    is_attack=True)
                        logging.info("--- {} epoch: {}, backdoor acc: {}".format(title, ep, backdoor_acc))
            # print execution time of federated training
            if self.args['time']:
                start_time = print_running_time('federated training', start_time)
            # save VFL
            if self.args['save_model']:
                self.vfl.save()

    def get_normal_representation_for_backdoor_label(self, test_loader):
        """
        collect normal representations output by the attacker's clean bottom model on normal inputs

        :param test_loader: testing dataset for main task
        :return: normal representations
        """
        attacker_id = 0
        self.vfl.set_eval()
        target_X, target_Y = None, None
        for batch_idx, (X, Y_batch, indices) in enumerate(test_loader):
            party_X_train_batch_dict = dict()
            if self.args['dataset'] != 'bhi':
                active_X_batch, Xb_batch = X
                if self.args['dataset'] == 'yahoo':
                    Xb_batch = Xb_batch.long()
                    Y_batch = Y_batch[0].long()
                party_X_train_batch_dict[0] = Xb_batch
            else:
                for i in range(self.args['n_passive_party']):
                    party_X_train_batch_dict[i] = X[:, i + 1:i + 2].squeeze(1)

            Y = Y_batch.numpy()
            target_indices = np.where(Y == self.args['backdoor_label'])[0]
            target_X = torch.cat([target_X, party_X_train_batch_dict[attacker_id][target_indices]], dim=0) \
                if target_X is not None else party_X_train_batch_dict[attacker_id][target_indices]
            if target_X.shape[0] >= 100:
                break
        result = self.vfl.party_dict[attacker_id].predict(target_X[:100])
        return result

    def lr_ba_attack(self, train_loader, test_loader,
                   backdoor_train_loader, backdoor_test_loader, backdoor_indices,
                   labeled_loader, unlabeled_loader, lr_ba_top_model=None,
                   add_from_unlabeled_tag=True, get_error_features_tag=True, random_tag=False):
        """
        conduct LR-BA attack, happens after VFL training

        :param train_loader: loader of normal train dataset
        :param test_loader: loader of normal test dataset
        :param backdoor_train_loader: loader of backdoor train dataset
        :param backdoor_test_loader: loader of backdoor test dataset
        :param backdoor_indices: indices of backdoor samples in normal train dataset
        :param labeled_loader: loader of labeled samples in normal train dataset
        :param unlabeled_loader: loader of unlabeled samples in normal train dataset
        :param lr_ba_top_model: inference head, not training if provided
        :param add_from_unlabeled_tag: whether to initialize from unlabeled samples
        :param get_error_features_tag: invalid
        :param random_tag: whether to randomly initialize backdoor latent representation
        :return: inference head
        """
        top_k = 1
        if self.dataset == 'cifar100':
            top_k = 5

        # compute main and backdoor task accuracy before conducting LR-BA
        logging.info("--- before LR-BA backdoor: main test acc: {}".
                     format(self.predict(test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'])))
        logging.info("--- before LR-BA backdoor: backdoor test acc: {}".
                     format(self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'],
                                         is_attack=True)))

        logging.info("--- LR-BA backdoor start")

        # conduct LR-BA attack
        bottom_model_list, lr_ba_top_model = lr_ba_backdoor(
            train_loader=train_loader,
            test_loader=test_loader,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            vfl=self.vfl,
            args=self.args,
            backdoor_train_loader=backdoor_train_loader,
            backdoor_test_loader=backdoor_test_loader,
            lr_ba_top_model=lr_ba_top_model,
            add_from_unlabeled_tag=add_from_unlabeled_tag,
            get_error_features_tag=get_error_features_tag,
            random_tag=random_tag
        )

        # compute main and backdoor task accuracy after conducting LR-BA if not evaluating execution time
        if not self.args['time']:
            for i, bottom_model in enumerate(bottom_model_list):
                self.vfl.party_dict[0].bottom_model = bottom_model
                if isinstance(self.args['lr_ba_generate_epochs'], list):
                    logging.info("--- LR-BA generate epochs: {}".format(self.args['lr_ba_generate_epochs'][i]))
                else:
                    logging.info("--- LR-BA generate epochs: {}".format(self.args['lr_ba_generate_epochs']))
                logging.info("--- after LR-BA backdoor: main test acc: {}".
                             format(self.predict(test_loader, num_classes=self.args['num_classes'],
                                                 dataset=self.args['dataset'], top_k=top_k,
                                                 n_passive_party=self.args['n_passive_party'])))
                logging.info("--- after LR-BA backdoor: backdoor test acc: {}".
                             format(self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                                 dataset=self.args['dataset'], top_k=top_k,
                                                 n_passive_party=self.args['n_passive_party'],
                                                 is_attack=True)))
        return lr_ba_top_model

    def baseline_attack(self, train_loader, test_loader,
                        backdoor_train_loader, backdoor_test_loader, backdoor_indices,
                        labeled_loader, unlabeled_loader, lr_ba_top_model=None):

        top_k = 1
        if self.dataset == 'cifar100':
            top_k = 5

        # compute main and backdoor task accuracy before conducting baseline attack
        logging.info("--- before baseline backdoor: main test acc: {}".
                     format(self.predict(test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'])))
        logging.info("--- before baseline backdoor: backdoor test acc: {}".
                     format(self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'],
                                         is_attack=True)))

        logging.info("--- baseline backdoor start")

        # conduct LR-BA attack
        bottom_model = baseline_backdoor(
            train_loader=train_loader,
            test_loader=test_loader,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            vfl=self.vfl,
            args=self.args,
            backdoor_train_loader=backdoor_train_loader,
            backdoor_test_loader=backdoor_test_loader,
            lr_ba_top_model=lr_ba_top_model
        )

        # compute main and backdoor task accuracy after conducting LR-BA
        self.vfl.party_dict[0].bottom_model = bottom_model
        logging.info("--- after baseline backdoor: main test acc: {}".
                     format(self.predict(test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'])))
        logging.info("--- after baseline backdoor: backdoor test acc: {}".
                     format(self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'],
                                         is_attack=True)))
        return

    def predict(self, test_loader, num_classes, dataset, top_k=1, n_passive_party=2, is_attack=False):
        """
        compute accuracy of VFL system on test dataset

        :param test_loader: loader of test dataset
        :param num_classes: number of dataset classes
        :param dataset: dataset name
        :param top_k: top-k accuracy
        :param n_passive_party: number of passive parties
        :param is_attack: whether to compute attack accuracy
        :return: accuracy
        """
        y_predict = []
        y_true = []
        with torch.no_grad():
            self.vfl.set_eval()
            for batch_idx, (X, targets, indices) in enumerate(test_loader):
                party_X_test_dict = dict()
                if dataset != 'bhi':
                    active_X_inputs, Xb_inputs = X
                    if self.args['dataset'] == 'yahoo':
                        active_X_inputs = active_X_inputs.long()
                        Xb_inputs = Xb_inputs.long()
                        targets = targets[0].long()
                    party_X_test_dict[0] = Xb_inputs
                else:
                    active_X_inputs = X[:, 0:1].squeeze(1)
                    for i in range(n_passive_party):
                        party_X_test_dict[i] = X[:, i+1:i+2].squeeze(1)
                y_true += targets.data.tolist()
                y_prob_preds = self.vfl.predict(active_X_inputs, party_X_test_dict)
                y_predict += y_prob_preds.tolist()
        acc = accuracy(y_true, y_predict, top_k=top_k, num_classes=num_classes, dataset=dataset, is_attack=is_attack)
        return acc

    def lr_ba_attack_for_representation(self, train_loader, test_loader,
                                      labeled_loader, unlabeled_loader, lr_ba_top_model=None,
                                      add_from_unlabeled_tag=True, get_error_features_tag=True, random_tag=False):
        """
        collect backdoor latent representation generated by LR-BA

        :param train_loader: loader of normal train dataset
        :param test_loader: loader of normal test dataset
        :param labeled_loader: loader of labeled samples in normal train dataset
        :param unlabeled_loader: loader of unlabeled samples in normal train dataset
        :param lr_ba_top_model: inference head, not training if provided
        :param add_from_unlabeled_tag: whether to initialize from unlabeled samples
        :param get_error_features_tag: invalid
        :param random_tag: whether to randomly initialize backdoor latent representation
        :return: inference head and backdoor latent representation
        """
        logging.info("--- LR-BA backdoor start")

        # conduct LR-BA attack
        lr_ba_top_model, representation = lr_ba_backdoor_for_representation(
            train_loader=train_loader,
            test_loader=test_loader,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            vfl=self.vfl,
            args=self.args,
            lr_ba_top_model=lr_ba_top_model,
            add_from_unlabeled_tag=add_from_unlabeled_tag,
            get_error_features_tag=get_error_features_tag,
            random_tag=random_tag
        )
        return lr_ba_top_model, representation[0]