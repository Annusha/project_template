#!/usr/bin/env python

""" Run experiments.
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'

import sys
sys.path.append('/BS/kukleva/work/project_template')

from dummy_data.dataloader_classification import DummyFeaturesDataset
import  dummy_data.update_arg_parse
from utils.arg_parse import opt
import dummy_model.model
import dummy_model.train
import dummy_model.test


def catch_inner(logger):
    train_dataset = DummyFeaturesDataset(mode='train')

    model, loss, optimizer = dummy_model.model.create_model(n_classes=train_dataset.n_classes)

    dummy_model.train.training(train_dataset,
                               model=model,
                               loss=loss,
                               optimizer=optimizer)


def pipeline(model_name, sfx):
    # update data-specific parameters
    if opt.dataset == 'dummy':
        update_arg_parse = dummy_data.update_arg_parse
    logger = update_arg_parse.update(model_name, sfx=sfx)
    catch_inner(logger)


def experiment1():
    model_name = 'exp1'
    sfx = str('sfx')
    pipeline(model_name, sfx=sfx)


if __name__ == '__main__':
    exec_idx = 0
    foo = {0: experiment1}

    foo[exec_idx]()
