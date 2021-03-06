#!/usr/bin/env python

""" Update / put the dataset dependant parameters here.
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'

import torch

from utils.logging_setup import setup_logger_path
from utils.arg_parse import opt


def update(model_name, sfx=''):
    # list all parameters which should be in the log name
    # map long names of parameters to short abbreviations for the log name
    args_map = {
        'epochs': 'ep%s',
        # 'model_name': '',
        # 'sfx': '',
        # 'dataset': '',
        # 'seed': '(%s)',
        'dropout': 'dp%s',
        'weight_decay': 'wd%s',
        'optim': '%s',
                }
    opt_d = vars(opt)
    opt_d['model_name'] = model_name
    opt_d['log_save_dir'] = opt.log_save_dir + '/%s' % opt.model_name
    opt_d['sfx'] = sfx
    opt_d['viz_env'] = '%s.%s_%s.' % (model_name, opt_d['dataset'], sfx)
    if torch.cuda.is_available():
        opt_d['device'] = 'cuda'
    else: opt_d['device'] = 'cpu'

    opt_d['i_dim'] = 65
    opt_d['data_path'] = '/BS/kukleva/work/project_template/dummy_data/features'
    opt_d['gt_file'] = '/BS/kukleva/work/project_template/dummy_data/gt.txt'

    log_name = ''
    # log_name_attr = sorted(args_map, key=lambda x: ('_' + str(x)) if x in ['model_name', 'sfx'] else x)
    log_name_attr = sorted(args_map)
    for attr_name in log_name_attr:
        attr = getattr(opt, attr_name)
        arg = args_map[attr_name]
        if isinstance(attr, bool):
            attr = arg if attr else '!' + arg
        else:
            attr = arg % str(attr)
        log_name += '%s.' % attr

    opt.log_name = log_name

    logger = setup_logger_path()

    # print in the log file all the set parameters
    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))
    return logger