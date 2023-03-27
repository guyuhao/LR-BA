# -*- coding: utf-8 -*-
from model.bhi.resnet_bhi import resnet_bhi
from model.bhi.top_model_fcn_bhi import top_model_fcn_bhi
from model.cifar.resnet_cifar import resnet_cifar
from model.cifar.top_model_fcn_cifar import top_model_fcn_cifar
from model.cinic.resnet_cinic import resnet_cinic
from model.cinic.top_model_fcn_cinic import top_model_fcn_cinic
from model.nus_wide.fcn_nus_wide import fcn_nus_wide
from model.nus_wide.top_model_fcn_nus_wide import top_model_fcn_nus_wide
from model.yahoo.bert_yahoo import bert_yahoo
from model.yahoo.top_model_fcn_yahoo import top_model_fcn_yahoo


def init_bottom_model(role, args):
    """
    initialize bottom model for parties

    :param role: party role, "active" or "passive"
    :param args: configuration
    :return: bottom model
    """
    param_dict = {
        'dataset': args['dataset'],
        'num_classes': args['num_classes'],
        'role': role,
        'cuda': args['cuda'],
        'pos': 'bottom'
    }
    # use active hyper-parameters for active party
    if role == 'active':
        param_dict['lr'] = args['active_bottom_lr']
        param_dict['momentum'] = args['active_bottom_momentum']
        param_dict['wd'] = args['active_bottom_wd']
        if args['active_bottom_stone'] is not None:
            param_dict['stone'] = args['active_bottom_stone']
            param_dict['gamma'] = args['active_bottom_gamma']
    # use passive hyper-parameters for passive party
    elif role == 'passive':
        param_dict['lr'] = args['passive_bottom_lr']
        param_dict['momentum'] = args['passive_bottom_momentum']
        param_dict['wd'] = args['passive_bottom_wd']
        if args['passive_bottom_stone'] is not None:
            param_dict['stone'] = args['passive_bottom_stone']
            param_dict['gamma'] = args['passive_bottom_gamma']

    # choose bottom model architecture according to dataset
    if args['dataset'] == 'nus_wide':
        if args['active_top_trainable']:
            output_dim = args['num_classes']
        else:
            output_dim = args['num_classes']
        if role == 'active':
            param_dict['input_dim'] = 634
        elif role == 'passive':
            param_dict['input_dim'] = 1000
        param_dict['output_dim'] = output_dim
        return fcn_nus_wide(param_dict=param_dict)
    elif 'cifar' in args['dataset']:
        # param_dict['optim'] = 'adam'
        if args['active_top_trainable']:
            output_dim = args['num_classes']
        else:
            output_dim = args['num_classes']
        param_dict['output_dim'] = output_dim
        return resnet_cifar(param_dict=param_dict)
    elif args['dataset'] == 'yahoo':
        if args['active_top_trainable']:
            output_dim = args['num_classes']
        else:
            output_dim = args['num_classes']
        param_dict['output_dim'] = output_dim
        return bert_yahoo(param_dict=param_dict)
    elif args['dataset'] == 'cinic':
        if args['active_top_trainable']:
            output_dim = args['num_classes']
        else:
            output_dim = args['num_classes']
        param_dict['output_dim'] = output_dim
        return resnet_cinic(param_dict=param_dict)
    elif args['dataset'] == 'bhi':
        if args['active_top_trainable']:
            output_dim = 5
        else:
            output_dim = args['num_classes']
        param_dict['output_dim'] = output_dim
        return resnet_bhi(param_dict=param_dict)


def init_top_model(args):
    """
    initialize top model for active party

    :param args: configuration
    :return: top model
    """
    param_dict = {
        'dataset': args['dataset'],
        'lr': args['active_top_lr'],
        'momentum': args['active_top_momentum'],
        'wd': args['active_top_wd'],
        'cuda': args['cuda'],
        'pos': 'top'
    }
    if args['active_top_stone'] is not None:
        param_dict['stone'] = args['active_top_stone']
        param_dict['gamma'] = args['active_top_gamma']

    # choose top model architecture according to dataset
    if args['dataset'] == 'nus_wide':
        param_dict['input_dim'] = 2*args['num_classes']
        param_dict['output_dim'] = args['num_classes']
        return top_model_fcn_nus_wide(param_dict=param_dict)
    elif 'cifar' in args['dataset']:
        param_dict['input_dim'] = 2 * args['num_classes']
        param_dict['output_dim'] = args['num_classes']
        return top_model_fcn_cifar(param_dict=param_dict)
    elif args['dataset'] == 'yahoo':
        param_dict['input_dim'] = 2 * args['num_classes']
        param_dict['output_dim'] = args['num_classes']
        return top_model_fcn_yahoo(param_dict=param_dict)
    elif args['dataset'] == 'cinic':
        param_dict['input_dim'] = 2 * args['num_classes']
        param_dict['output_dim'] = args['num_classes']
        return top_model_fcn_cinic(param_dict=param_dict)
    elif args['dataset'] == 'bhi':
        param_dict['input_dim'] = (args['n_passive_party']+1) * 5
        param_dict['output_dim'] = args['num_classes']
        return top_model_fcn_bhi(param_dict=param_dict)
