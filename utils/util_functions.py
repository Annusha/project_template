#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'


import os.path as ops
import torch
import time

from utils.logging_setup import logger
from utils.arg_parse import opt


class Meter(object):
    def __init__(self, mode='', name=''):
        self.mode = mode
        self.name = name
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def log(self):
        logger.debug('%s %s: %f' % (self.mode.upper(), self.name, self.avg))

    def viz_dict(self):
        return {
            '%s/%s' % (self.name, self.mode.upper()): self.avg
        }

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val * n
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def join_data(data1, data2, f):
    """Simple use of numpy functions vstack and hstack even if data not a tuple
    Args:
        data1 (arr): array or None to be in front of
        data2 (arr): tuple of arrays to join to data1
        f: vstack or hstack from numpy
    Returns:
        Joined data with provided method.
    """
    if isinstance(data1, torch.Tensor):
        data1 = data1.numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.numpy()
    if isinstance(data2, tuple):
        data2 = f(data2)
    if data2 is None:
        data2 = data1
    elif data1 is not None:
        data2 = f((data1, data2))
    return data2


def adjust_lr(optimizer, lr):
    """Decrease learning rate by 0.1 during training"""
    lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def load_model(name='mlp_text'):
    if opt.resume_str:
        resume_str = opt.resume_str
    else:
        resume_str = '%s.pth.tar' % opt.log_name
    opt.resume_str = resume_str
    if opt.device == 'cpu':
        checkpoint = torch.load(ops.join(opt.store_root, 'models', name, resume_str),
                                map_location='cpu')
    else:
        checkpoint = torch.load(ops.join(opt.store_root, 'models', name, resume_str))
    checkpoint = checkpoint['state_dict']
    print('loaded model: ' + ' %s' % resume_str)
    return checkpoint


def load_optimizer():
    if opt.resume_str:
        resume_str = opt.resume_str
    else:
        resume_str = '%s.pth.tar' % opt.log_name
    opt.resume_str = resume_str
    if opt.device == 'cpu':
        checkpoint = torch.load(ops.join(opt.store_root, 'models', opt.model_name, resume_str),
                                map_location='cpu')
    else:
        checkpoint = torch.load(ops.join(opt.store_root, 'models', opt.model_name, resume_str))
    checkpoint = checkpoint['optimizer']
    print('loaded optimizer')
    return checkpoint


def timing(f):
    """Wrapper for functions to measure time"""
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('%s took %0.3f ms ~ %0.3f min ~ %0.3f sec'
                     % (f, (time2-time1)*1000.0,
                        (time2-time1)/60.0,
                        (time2-time1)))
        return ret
    return wrap
