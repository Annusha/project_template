#!/usr/bin/env python

""" File with the dummy model with the template training/testing functions.
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.arg_parse import opt
from utils.logging_setup import logger


class DummyModel(nn.Module):
    # classification task into 'n_classes'
    # 1 layer MLP
    def __init__(self, n_classes):
        super(DummyModel, self).__init__()
        self.n_classes = n_classes

        self.fc = nn.Linear(opt.i_dim, n_classes)

    def forward(self, input):
        x = input['features'].float()
        x = x.to(device=opt.device, non_blocking=True)
        '''
        non_blocking
        If the next operation depends on your data, you won’t notice any speed advantage. 
        However, if the asynchronous data transfer is possible, you might hide the transfer 
        time in another operation.
        use with pin_memory=True only
        '''
        x = self.fc(x)
        return {'probs': x.squeeze()}


class DummyLoss(nn.Module):
    def __init__(self):
        super(DummyLoss, self).__init__()

    def forward(self, output, input):
        probs = output['probs'].to(opt.device)
        target = input['labels'].to(opt.device)

        return F.cross_entropy(probs, target)


def create_model(**kwargs):
    model = DummyModel(n_classes=kwargs['n_classes'])
    loss = DummyLoss()
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.lr,
                                     weight_decay=opt.weight_decay)

    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)

    logger.debug(str(model))
    for name, param in model.named_parameters():
        logger.debug('%s\n%s' % (str(name), str(param.norm())))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer
