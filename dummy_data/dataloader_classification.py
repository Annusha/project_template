#!/usr/bin/env python

""" Dataset for the classification of the dummy data.
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'


from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import os.path as ops
import numpy as np

from utils.arg_parse import opt
from utils.logging_setup import logger


# additional subclass to load features
# class DummyFeatures:
#     def __init__(self):
#         pass
#


class DummyFeaturesDataset(Dataset):
    def __init__(self, mode='train'):
        logger.debug('Load dummy features. %s mode.' % mode.upper())
        self.features = {}
        self.idx2cl = {}
        self.cl2idx = {}
        self.labels = {}  # sample2clidx
        f_read = None
        with open(opt.gt_file, 'r') as f:
            for idx, line in tqdm(enumerate(f)):
                filename, cl = line.strip().split()
                if f_read is None:
                    if filename.endswith('npy'): f_read = np.load
                    else: f_read = np.loadtxt
                # self.idx2cl[idx] = cl
                if cl not in self.cl2idx:
                    self.cl2idx[cl] = len(self.cl2idx)
                    self.idx2cl[len(self.cl2idx)-1] = cl
                self.labels[idx] = self.cl2idx[cl]
                self.features[idx] = f_read(ops.join(opt.data_path, filename)).mean(axis=0)

        self.n_classes = len(self.cl2idx)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, sample_idx):
        return {'features': self.features[sample_idx], 'labels': self.labels[sample_idx]}
