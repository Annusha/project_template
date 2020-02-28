#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'

import torch


from utils.plotting_utils import visdom_plot_losses
from utils.util_functions import Meter
from utils.logging_setup import viz
from utils.eval import Precision
from utils.arg_parse import opt


def testing(test_dataset, model, loss, epoch=1, mode='val', time_id=''):
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.num_workers,
                                              drop_last=False)

    meter = Meter(mode=mode, name='loss')
    model.eval()
    prec = Precision(mode)
    with torch.no_grad():
        for idx, input in enumerate(test_loader):
            labels = input['labels']
            if len(labels) == 1: raise EnvironmentError('LAZY ANNA')
            output = model(input)
            loss_values = loss(output, input)
            meter.update(loss_values.item(), len(labels))
            prec.update_probs(pr_probs=output['probs'], gt=labels)

    meter.log()
    prec.log()

    if opt.viz:
        visdom_plot_losses(viz.env, opt.log_name + '-loss-' + time_id, epoch,
                           xylabel=('epoch', 'loss'), **meter.viz_dict())
        visdom_plot_losses(viz.env, opt.log_name + '-prec-' + time_id, epoch,
                           xylabel=('epoch', 'prec'), **prec.viz_dict())

    return {'top1': prec.top1()}