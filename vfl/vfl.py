import logging

from model.init_model import init_bottom_model, init_top_model
from vfl.backdoor.g_r_passive_model import GRPassiveModel
from vfl.party_models import VFLActiveModel, VFLPassiveModel


class VFL(object):
    """
    VFL system
    """
    def __init__(self, active_party):
        super(VFL, self).__init__()
        self.active_party = active_party
        self.party_dict = dict()  # passive parties dict
        self.party_ids = list()  # id list of passive parties
        self.is_debug = False

    def set_debug(self, is_debug):
        self.is_debug = is_debug

    def add_party(self, *, id, party_model):
        """
        add passive party

        :param id: passive party id
        :param party_model: passive party
        """
        self.party_dict[id] = party_model
        self.party_ids.append(id)

    def set_current_epoch(self, ep):
        """
        set current train epoch

        :param ep: current train epoch
        """
        self.active_party.set_epoch(ep)
        for i in self.party_ids:
            self.party_dict[i].set_epoch(ep)

    def fit(self, active_X, y, party_X_dict, indices):
        """
        VFL training in one batch

        :param active_X: features of active party
        :param y: labels
        :param dict party_X_dict: features of passive parties, the key is passive party id
        :param indices: indices of samples in current batch
        :return: loss
        """
        if self.is_debug:
            logging.info("==> start fit")

        # set features and labels for active party
        self.active_party.set_batch(active_X, y)
        self.active_party.set_indices(indices)

        # set features for all passive parties
        for idx, party_X in party_X_dict.items():
            self.party_dict[idx].set_batch(party_X, indices)

        # all passive parties output latent representations and upload them to active party
        comp_list = []
        for id in self.party_ids:
            party = self.party_dict[id]
            logits = party.send_components()
            comp_list.append(logits)
        self.active_party.receive_components(component_list=comp_list)

        # active party compute gradients based on labels and update parameters of its bottom model and top model
        self.active_party.fit()
        loss = self.active_party.get_loss()

        # active party send gradients to passive parties, then passive parties update parameters of their bottom model
        parties_grad_list = self.active_party.send_gradients()
        for index, id in enumerate(self.party_ids):
            party = self.party_dict[id]
            party.receive_gradients(parties_grad_list[index])

        return loss

    def save(self):
        """
        save all models in VFL, including top model and all bottom models
        """
        self.active_party.save()
        for id in self.party_ids:
            self.party_dict[id].save()

    def load(self, load_attack=False):
        """
        load all models in VFL, including top model and all bottom models

        :param load_attack: invalid
        """
        self.active_party.load()
        for id in self.party_ids:
            if load_attack and id == 0:
                self.party_dict[id].load(load_attack=True)
            else:
                self.party_dict[id].load()

    def predict(self, active_X, party_X_dict, attack_output=None):
        """
        predict label with help of all parties

        :param active_X: features of active party
        :param dict party_X_dict: features of passive parties, the key is passive party id
        :param attack_output: latent represent output by the attacker if provided, otherwise the attacker output using bottom model
        :return: prediction label
        """
        comp_list = []
        # passive parties send latent representations
        for id in self.party_ids:
            # if attack_output is provided, the attacker adopts attack_output as its latent representation without using bottom model
            if attack_output is not None and id == 0:
                comp_list.append(attack_output)
            else:
                comp_list.append(self.party_dict[id].predict(party_X_dict[id]))

        # active party make the final prediction
        return self.active_party.predict(active_X, component_list=comp_list)

    def set_train(self):
        """
        set train mode for all parties
        """
        self.active_party.set_train()
        for id in self.party_ids:
            self.party_dict[id].set_train()

    def set_eval(self):
        """
        set eval mode for all parties
        """
        self.active_party.set_eval()
        for id in self.party_ids:
            self.party_dict[id].set_eval()

    def scheduler_step(self):
        """
        adjust learning rate for all parties during training
        """
        self.active_party.scheduler_step()
        for id in self.party_ids:
            self.party_dict[id].scheduler_step()

    def zero_grad(self):
        """
        clear gradients for all parties
        """
        self.active_party.zero_grad()
        for id in self.party_ids:
            self.party_dict[id].zero_grad()


def get_vfl(args, backdoor_indices=None, backdoor_target_indices=None, train_loader=None):
    """
    generate VFL system

    :param args: configuration
    :param backdoor_indices: indices of backdoor samples in normal train dataset, used by gradient-replacement
    :param backdoor_target_indices: indices of samples labeled backdoor class in normal train dataset, used by gradient-replacement
    :return: VFL system
    """
    # build bottom model for active party
    active_bottom_model = init_bottom_model('active', args)

    # build bottom model for passive parties
    party_model_list = list()
    for i in range(0, args['n_passive_party']):
        passive_party_model = init_bottom_model('passive', args)
        party_model_list.append(passive_party_model)

    # build top model for active party
    active_top_model = None
    if args['active_top_trainable']:
        active_top_model = init_top_model(args)

    # generate active party
    active_party = VFLActiveModel(bottom_model=active_bottom_model,
                                  args=args,
                                  top_model=active_top_model)

    # generate passive parties
    party_list = list()
    for i, model in enumerate(party_model_list):
        if args['backdoor'] == 'g_r':
            # generate malicious passive party for gradient-replacement backdoor, only support one passive party
            passive_party = GRPassiveModel(bottom_model=model,
                                           amplify_ratio=args['g_r_amplify_ratio'],
                                           top_trainable=args['active_top_trainable'])

            backdoor_X = dict()
            if train_loader is not None:
                for X, _, indices in train_loader:
                    temp_indices = list(set(backdoor_indices) & set(indices.tolist()))
                    if len(temp_indices) > 0:
                        if args['dataset'] != 'bhi':
                            _, Xb_batch = X
                            if args['dataset'] == 'yahoo':
                                Xb_batch = Xb_batch.long()
                        else:
                            Xb_batch = X[:, 1:2].squeeze(1)
                        for temp in temp_indices:
                            backdoor_X[temp] = Xb_batch[indices.tolist().index(temp)]
            passive_party.set_backdoor_indices(backdoor_target_indices, backdoor_indices, backdoor_X)
        else:
            passive_party = VFLPassiveModel(bottom_model=model, id=i)
        party_list.append(passive_party)

    vfl = VFL(active_party)
    for index, party in enumerate(party_list):
        vfl.add_party(id=index, party_model=party)
    vfl.set_debug(is_debug=False)
    return vfl
