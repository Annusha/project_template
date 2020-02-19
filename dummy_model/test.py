#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'

import torch
import numpy as np
import os.path as ops
from collections import defaultdict


from utils.arg_parse import opt
from utils.util_functions import Meter, dir_check
from utils.eval import Precision
from utils.logging_setup import logger


def testing(test_dataset, model, loss, epoch=1, mode='val'):
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.num_workers,
                                              drop_last=False)

    meter = Meter()
    model.eval()
    prec = Precision()
    with torch.no_grad():
        for idx, input in enumerate(test_loader):
            labels = input['labels']
            if len(labels) == 1: raise EnvironmentError('LAZY ANNA')
            output = model(input)
            loss_values = loss(output, input)
            meter.update(loss_values.item(), len(labels))
            prec.update_probs(pr_probs=output['probs'], gt=labels)

    logger.debug('total items during test: %d' % prec.total)

    logger.debug('%s loss: %f' % (mode.upper(), meter.avg))
    logger.debug('%s pr@1: %f' % (mode.upper(), prec.top1()))
    logger.debug('%s pr@5: %f' % (mode.upper(), prec.top5()))


    # add visualisation here!!!!!

    return {'top1': prec.top1()}