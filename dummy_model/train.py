#!/usr/bin/env python

""" File with the training loop.
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'

import torch.backends.cudnn as cudnn
from datetime import datetime
from os.path import join
import numpy as np
import random
import torch
import time
import copy

from utils.util_functions import dir_check, adjust_lr, Meter
from utils.model_saver import ModelSaver
from utils.logging_setup import logger
from dummy_model.test import testing
from utils.arg_parse import opt


def training(train_dataset, **kwargs):
    train_start_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    logger.debug('Set parameters and the model. Start training time is: %s' % train_start_time)

    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']

    batch_time = Meter()
    data_time = Meter()
    loss_meter = Meter()

    # do not modify opt.lr, use everywhere here adjustible_lr
    adjustable_lr = opt.lr
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers,
                                                   drop_last=False)

    logger.debug('Starting training for %d epochs:' % opt.epochs)

    model_saver_val = ModelSaver(path=join(opt.storage, 'models', opt.model_name, opt.log_name, 'val'))
    # model_saver_test = ModelSaver(path=join(opt.storage, 'models', kwargs['name'], opt.log_name, 'test'))
    # if you need test set validation, add corresponding functions below similarly to val set

    for epoch in range(opt.epochs):
        model.to(opt.device)
        model.train()
        train_dataset.epoch = epoch

        logger.debug('Epoch # %d' % epoch)

        # adjust learning rate if necessary
        if epoch and epoch % 50 == 0:  # every 50 epochs
            adjustable_lr = adjust_lr(optimizer, adjustable_lr)

        if epoch == 0:
            testing(train_dataset, model, loss, epoch=0, mode='train')

        end = time.time()

        n_train_samples = 0
        for i, input in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            labels = input['labels']

            if len(labels) == 1: raise EnvironmentError('LAZY ANNA')

            output = model(input)
            loss_values = loss(output, input)
            loss_meter.update(loss_values.item(), len(labels))

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            n_train_samples += len(labels)

            if i % opt.debug_freq == 0 and i:
                logger.debug('Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_dataloader), batch_time=batch_time, data_time=data_time, loss=loss_meter))

        logger.debug('Number of training sampler within one epoch: %d' % n_train_samples)
        logger.debug('Loss: %f' % loss_meter.avg)
        loss_meter.reset()



        if opt.test_freq and epoch % opt.test_freq == 0:
            # test the model every opt.test_freq epoch
            testing(train_dataset, model, loss, epoch=epoch, mode='train')

        if opt.save_model and epoch % opt.save_model == 0:
            if opt.test_val:
                check_val = testing(kwargs['val_dataset'], model, loss, epoch=epoch, mode='val')
                if model_saver_val.check(check_val):
                    save_dict = {'epoch': epoch,
                                 'state_dict': copy.deepcopy(model.state_dict()),
                                 'optimizer': copy.deepcopy(optimizer.state_dict().copy())}

            logger.debug(opt.log_name)

        if opt.save_model:
            model_saver_val.save()

    # save the last checkpoint of the training
    if opt.save_model:
        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        dir_check(join(opt.storage, 'models', opt.model_name, opt.log_name))
        torch.save(save_dict,
                   join(opt.storage, 'models', opt.model_name, opt.log_name, 'last_%d.pth.tar' % epoch))

    return model