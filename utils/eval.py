#!/usr/bin/env python

""" Classes for metric calculations.
"""

__author__ = 'Anna Kukleva'
__date__ = 'February 2020'


import numpy as np
import warnings
import torch

from utils.logging_setup import logger


class Precision(object):
    def __init__(self, mode=''):
        self.mode = mode
        self._top1 = 0
        self._top5 = 0
        self.total = 0

        self.preds = []
        self.output = dict()

    def log(self):
        logger.debug('total items during test: %d' % self.total)
        logger.debug('%s pr@1: %f' % (self.mode.upper(), self.top1()))
        logger.debug('%s pr@5: %f' % (self.mode.upper(), self.top5()))

    def viz_dict(self):
        return {
            'pr@1/%s' % self.mode.upper(): self.top1(),
            'pr@5/%s' % self.mode.upper(): self.top5(),
        }

    def _to_numpy(self, a):
        torch_types = []
        torch_types.append(torch.Tensor)
        torch_types.append(torch.nn.Parameter)
        if isinstance(a, list):
            return np.array(a)
        if len(torch_types) > 0:
            if isinstance(a, torch.autograd.Variable):
                # For PyTorch < 0.4 comptability.
                warnings.warn(
                    "Support for versions of PyTorch less than 0.4 is deprecated "
                    "and will eventually be removed.", DeprecationWarning)
                a = a.data
        for kind in torch_types:
            if isinstance(a, kind):
                # For PyTorch < 0.4 comptability, where non-Variable
                # tensors do not have a 'detach' method. Will be removed.
                if hasattr(a, 'detach'):
                    a = a.detach()
                return a.cpu().numpy()
        return a

    def update_probs(self, pr_probs=None, gt=None, pr_classes=None, **kwargs):
        '''
        Args:
            pr_probs: None or matrix (batch_size, N_classes) with probabilities (confidence) of the model to correspond
                sample to one of the classes
            gt: vector (batch_size) with ground truth assignments
            pr_classes: if as input there is a sorted matrix (batch_size, N_classes) where the first column correspond to
             the most probable class for the relative sample
            **kwargs: whatever else can be included as additional parameter

        Returns: all additional return values should be written into self.output['key']

        '''
        self.total += len(gt)
        gt = self._to_numpy(gt).squeeze()  # make sure that it's vector now, and not the matrix
        if pr_classes is None:
            assert len(pr_probs) == len(gt)
            pr_probs = self._to_numpy(pr_probs)
            pr_classes = np.argsort(-pr_probs, axis=1)
        else:
            pr_classes = self._to_numpy(pr_classes)
            assert len(pr_classes) == len(gt)

        self._top1 += np.sum((pr_classes[:, 0] == gt))
        self._top5 += np.sum([1 for top5_classes, gt_class in zip(pr_classes[:, :5], gt) if gt_class in top5_classes])

        # to have an output confusion matrix
        try:
            conf_mat = kwargs['conf_mat']
            for gt_label, pr_label in zip(gt, pr_classes[:, 0]):
                conf_mat[gt_label, pr_label] += 1
            self.output['conf_mat'] = conf_mat
        except KeyError: pass

    def top1(self):
        return self._top1 / self.total

    def top5(self):
        return self._top5 / self.total