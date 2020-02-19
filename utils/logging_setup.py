#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'


import torch.backends.cudnn as cudnn
import os.path as ops
import numpy as np
import datetime
import logging
import random
import torch
import sys
import os
import re

from utils.arg_parse import opt


logger = logging.getLogger('basics')
logger.setLevel(opt.log_mode)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

filename = sys.argv[0]
search = re.search(r'\/*(\w*).py', filename)
filename = search.group(1)

def setup_logger_path():
    global logger, viz

    # make everything deterministic -> seed setup everywhere
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False
    '''
    Benchmark mode is good whenever your input sizes for your network do not vary. 
    This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time).
    This usually leads to faster runtime. But if your input sizes changes at each iteration, then cudnn will benchmark 
    every time a new size appears, possibly leading to worse runtime performances.
    '''

    # check if log folder exists
    os.makedirs(opt.log_save_dir + '/%s' % opt.model_name, exist_ok=True)
    # if not ops.exists(opt.log_save_dir):
    #     os.mkdir(opt.log_save_dir)

    path_logging = ops.join(opt.log_save_dir, '%s_%s(%s)' % (opt.log_name,
                                                             filename,
                                                             str(datetime.datetime.now()).split('.')[0]))

    # close previously openned logging files if exist:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHadnler):
            handler.close()
            logger.handlers.remove(handler)

    fh = logging.FileHandler(path_logging, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelno)s - %(filename)s - '
                                  '%(funcName)s - %(message)s')

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger



