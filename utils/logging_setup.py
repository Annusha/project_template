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

    try:
        # check if log folder exists
        os.makedirs(opt.log_save_dir + '_0')
        opt.log_save_dir += '_0'
    except FileExistsError:
        rootdir, subdir = ops.split(opt.log_save_dir)
        idx = 0
        # check each folder with logs, if it contains for than 100 logs ->
        # create a new folder with incremented index _idx
        for subdirname in sorted(os.listdir(rootdir)):
            if subdir == re.search(r'(.*)_\d*', subdirname).group(1):
                subdir_idx = int(re.search(r'.*_(\d)*', subdirname).group(1))
                if subdir_idx >= idx:
                    if len(os.listdir(ops.join(rootdir, subdirname))) > 100:
                        idx = subdir_idx + 1
                    else:
                        idx = subdir_idx
        opt.log_save_dir += '_%d' % idx
        os.makedirs(opt.log_save_dir, exist_ok=True)

    # check how many files already there
    # if len(os.listdir(opt.log_save_dir)) > 100:



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



