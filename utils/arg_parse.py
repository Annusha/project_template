#!/usr/bin/env python

""" Main parameters for the project. Other parameters are dataset specific.
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'


import argparse

parser = argparse.ArgumentParser()

#######################################################################################
### DATA
parser.add_argument('--dataset', default='dummy',
                    help='dummy: artificial dataset as an example')
parser.add_argument('--data_path', default='')
parser.add_argument('--gt_file', default='')


#######################################################################################
### STORAGE
parser.add_argument('--storage', default='/BS/kukleva/work/storage')
parser.add_argument('--save_model', default=0, type=int,
                    help='how often save model, 0 - do not save at all')

#######################################################################################
### GENERAL HYPERPARAMS
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=3e-3, type=float)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=1)
parser.add_argument('--device', default='cuda',
                    help='cuda | cpu')
parser.add_argument('--optim', default='adam',
                    help='optimizer for the network: adam | sgd')

#######################################################################################
### VALIDATION & TEST
parser.add_argument('--test_freq', default=1, type=int)

#######################################################################################
### LOGS
parser.add_argument('--log_mode', default='DEBUG',
                    help='DEBUG | INFO | WARNING | ERROR | CRITICAL')
parser.add_argument('--log_save_dir', default='/BS/kukleva/nobackup/logs')
parser.add_argument('--log_name', default='')
parser.add_argument('--debug_freq', default=1, type=int)


#######################################################################################
### W & B



opt = parser.parse_args()