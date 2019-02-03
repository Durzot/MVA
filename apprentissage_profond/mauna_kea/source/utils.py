# -*- coding: utf-8 -*-
"""
Usefuls functions for other parts of the project
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""
import os
import random
import numpy as np

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def adjust_learning_rate(optimizer, epoch, phase):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if (epoch%phase==(phase-1)):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/10.

class AccuracyValueMeter(object):
    """Computes and stores the accuracy of predictions per category"""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.val = {i: [0 for _ in range(self.n_classes)] for i in range(self.n_classes)}
        self.avg = {i: [0. for _ in range(self.n_classes)] for i in range(self.n_classes)}
        self.sum = {i: [0 for _ in range(self.n_classes)] for i in range(self.n_classes)}
        self.sum_acc = 0
        self.acc = 0
        self.count = 0

    def update(self, pred, label, size=1):
        self.val = {i: [0 for _ in range(self.n_classes)] for i in range(self.n_classes)}
        for p, l in zip(pred, label):
            self.val[p][l] += 1
            self.sum[p][l] += 1
        for i in range(self.n_classes):
            if sum(self.sum[i]) == 0:
                self.avg[i] = [0 for _ in range(self.n_classes)]
            else:
                self.avg[i] = [x/sum(self.sum[i]) for x in self.sum[i]]
        self.sum_acc += np.sum(pred==label)
        self.count += size
        self.acc = self.sum_acc/self.count

