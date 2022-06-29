#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import numpy as np
import torch


class iouEval:
    def __init__(self, n_classes, device, ignore=None):
        self.n_classes = n_classes
        self.device = device
        # if ignore is larger than n_classes, consider no ignoreIndex
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).long()
        print("[IOU EVAL] IGNORE: ", self.ignore)
        print("[IOU EVAL] INCLUDE: ", self.include)
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device).long()
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used

    def addBatch(self, x, y):  # x=preds, y=targets idem: argmax, proj_labels
        # if numpy, pass to pytorch
        # to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # torch.Size[batch_size x H x W]
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        # torch.Size[2, batch_size x H x W]
        idxs = torch.stack([x_row, y_row], dim=0)

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            # torch.Size[batch_size x H x W]
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (rows = pred, cols = gt)
        # tensor.index_put_(indices, values, accumulate=True) is equivalent to tensor[indices] += values
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True)

    def getStats(self):
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone().double()
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        # get the clean stats
        # (rows = pred, cols = gt)
        tp = conf.diag()          # true positive
        fp = conf.sum(dim=1) - tp # false positive || over all columns  == for each row
        fn = conf.sum(dim=0) - tp # false negative || over all rows == for each column
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        # print(tp.shape)
        # print(fp.shape)
        # print(fn.shape)
        # torch.Size[self.n_classes]
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        # returns "iou mean", "iou per class" ALL CLASSES idem: jaccard, class_jaccard
        return iou_mean, iou

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"
