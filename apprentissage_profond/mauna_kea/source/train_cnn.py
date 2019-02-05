# -*- coding: utf-8 -*-
"""
Functions to train models on the train set
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import os
import sys
import argparse
import numpy as np 
import pandas as pd
import time
import datetime

import torch
from sklearn import metrics
import json

sys.path.append("./source")
from dataset import *
from models import *
from utils import *

def get_time():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M") 

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--n_classes', type=int, help='number of classes', default=4)
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--st_epoch', type=int, default=0, help='if continuing training, epoch from which to continue')
parser.add_argument('--model', type=str, default=None,  help='optional reload model path')
parser.add_argument('--model_name', type=str, default='benchmark',  help='name of the model for log')
parser.add_argument('--criterion', type=str, default='cross_entropy',  help='name of the criterion to use')
parser.add_argument('--optimizer', type=str, default='adam',  help='name of the optimizer to use')
parser.add_argument('--lr', type=float, default=1e-3,  help='learning rate')
parser.add_argument('--cuda', type=int, default=0, help='set to 1 to use cuda')
opt = parser.parse_args()

dataset_train = MaunaKea(train=True)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

dataset_test = MaunaKea(train=False)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

print('training set size %d' % len(dataset_train))
print('test set size %d' % len(dataset_test))

n_batch = len(dataset_train)//opt.batch_size

# ========================== NETWORK AND OPTIMIZER ========================== #
network = MaunaNet4(n_classes=opt.n_classes)
network.apply(init_weight)

if opt.model is not None:
    network.load_state_dict(torch.load(opt.model))
    print("Weights from %s loaded" % opt.model)

if opt.cuda:
    network.cuda()

if opt.optimizer == "adam":
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr)
elif opt.optimizer == "sgd":
    optimizer = torch.optim.SGD(network.parameters(), lr=opt.lr)
else:
    raise ValueError("Please choose between 'adam' and 'sgd' for the --optimizer")

if opt.criterion == "cross_entropy":
    criterion = torch.nn.CrossEntropyLoss()
else:
    raise ValueError("Please choose 'cross_entropy' for --criterion")

# ====================== DEFINE STUFF FOR LOGS ====================== #
log_path = os.path.join('log', 'cnn')
if not os.path.exists(log_path):
    os.mkdir(log_path)

log_file = os.path.join(log_path, 'cnn_%s.txt' % opt.model_name)
with open(log_file, 'a') as log:
    log.write(str(network) + '\n')

value_meter_train = AccuracyValueMeter(opt.n_classes)
value_meter_test = AccuracyValueMeter(opt.n_classes)

log_train_file = "./log/cnn/logs_train_%s.csv" % opt.model_name
log_test_file = "./log/cnn/logs_test_%s.csv" % opt.model_name

if not os.path.exists(log_train_file): 
    df_logs_train = pd.DataFrame(columns=['model', 'epoch', 'n_epoch', 'date', 'loss', 'acc', 'lr', 'optim', 'crit',
                                          'pred_0_0', 'pred_0_1', 'pred_0_2', 'pred_0_3',
                                          'pred_1_0', 'pred_1_1', 'pred_1_2', 'pred_1_3', 
                                          'pred_2_0', 'pred_2_1', 'pred_2_2', 'pred_2_3', 
                                          'pred_3_0', 'pred_3_1', 'pred_3_2', 'pred_3_3'])
else:
    df_logs_train = pd.read_csv(log_train_file, header='infer')

if not os.path.exists(log_test_file):
    df_logs_test = pd.DataFrame(columns=['model', 'epoch', 'n_epoch', 'date', 'loss', 'acc', 'lr', 'optim', 'crit'
                                         'pred_0_0', 'pred_0_1', 'pred_0_2', 'pred_0_3',
                                         'pred_1_0', 'pred_1_1', 'pred_1_2', 'pred_1_3', 
                                         'pred_2_0', 'pred_2_1', 'pred_2_2', 'pred_2_3', 
                                         'pred_3_0', 'pred_3_1', 'pred_3_2', 'pred_3_3'])
else:
    df_logs_test = pd.read_csv(log_test_file, header='infer')

# ====================== LEARNING LOOP ====================== #
for epoch in range(opt.st_epoch, opt.n_epoch):
    # TRAINING
    st_time = time.time()
    network.train()
    value_meter_train.reset()
    loss_train = 0
    
    for batch, (data, label) in enumerate(loader_train):
        if opt.cuda:
            data, label = data.cuda(), label.cuda()
    
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, label)
        loss_train += loss
        loss.backward()
        optimizer.step()
    
        pred = output.cpu().data.numpy().argmax(axis=1)
        label = label.cpu().data.numpy()
        value_meter_train.update(pred, label, opt.batch_size)
    
        print('[train epoch %d/%d ; batch: %d/%d] train loss: %.3g' % (epoch+1, opt.n_epoch, batch+1, n_batch, loss.item()))
        
    dt = time.time()-st_time
    s_time = "%d min %d sec" % (dt//60, dt%60)
    
    print('='*40)
    print('[train epoch %d/%d] | loss %.3g | nw acc %.3g | time %s' % (epoch+1, opt.n_epoch, loss_train, value_meter_train.acc, s_time))
    for i in range(opt.n_classes):
        print('cat %d: %s and %s' % (i, value_meter_train.sum[i], [float("{0:0.4f}".format(f)) for f in value_meter_train.avg[i]]))
    print('='*40)
    
    with open(log_file, 'a') as log:
        log.write('[train epoch %d/%d] | loss %.3g | nw acc %.3g | time %s' % (epoch+1, opt.n_epoch, loss_train, value_meter_train.acc, s_time) +  '\n')
        for i in range(opt.n_classes):
            log.write('cat %d: %s and %s' % (i, value_meter_train.sum[i], [float("{0:0.4f}".format(f)) for f in value_meter_train.avg[i]]) + '\n')

    row_train = {'model': opt.model_name, 
                 'epoch': epoch+1, 
                 'n_epoch': opt.n_epoch, 
                 'date': get_time(), 
                 'loss': loss_train, 
                 'acc': value_meter_train.acc,
                 'lr': opt.lr,
                 'optim': opt.optimizer,
                 'crit': opt.criterion}

    for i in range(opt.n_classes):
        for j in range(opt.n_classes):
            row_train["pred_%d_%d" % (i, j)] = value_meter_train_sum[i][j]
    
    df_logs_train = df_logs_train.append(row_train, ignore_index=True)
    df_logs_train.to_csv(log_train_file, header=True, index=False)

    # TEST
    st_time = time.time()
    network.eval()
    value_meter_test.reset()
    loss_test = 0
    
    with torch.no_grad():
        for batch, (data, label) in enumerate(loader_test):
            if opt.cuda:
                data, label = data.cuda(), label.cuda()
            
            output = network(data)
            loss = criterion(output, label)
            loss_test += loss
    
            pred = output.cpu().data.numpy().argmax(axis=1)
            label = label.cpu().data.numpy()
            value_meter_test.update(pred, label, opt.batch_size)
    
    dt = time.time()-st_time
    s_time = "%d min %d sec" % (dt//60, dt%60)

    print('='*40)
    print('[test epoch %d/%d] | loss %.3g | nw acc %.3g | time %s' % (epoch+1, opt.n_epoch, loss_test, value_meter_test.acc, s_time))
    for i in range(opt.n_classes):
        print('cat %d: %s and %s' % (i, value_meter_test.sum[i], [float("{0:0.4f}".format(f)) for f in value_meter_test.avg[i]]))
    print('='*40)
    
    with open(log_file, 'a') as log:
        log.write('[test epoch %d/%d] | loss %.3g | nw acc %.3g | time %s' % (epoch+1, opt.n_epoch, loss_test, value_meter_test.acc, s_time) + '\n')
        for i in range(opt.n_classes):
            log.write('cat %d: %s and %s' % (i, value_meter_test.sum[i], [float("{0:0.4f}".format(f)) for f in value_meter_test.avg[i]]) + '\n')

    row_test = {'model': opt.model_name, 
                'epoch': epoch+1, 
                'n_epoch': opt.n_epoch, 
                'date': get_time(), 
                'loss': loss_test, 
                'acc': value_meter_test.acc
                'lr': opt.lr,
                'optim': opt.optimizer,
                'crit': opt.criterion}

    for i in range(opt.n_classes):
        for j in range(opt.n_classes):
            row_test["pred_%d_%d" % (i, j)] = value_meter_tset_sum[i][j]
    
    df_logs_test = df_logs_test.append(row_test, ignore_index=True)
    df_logs_test.to_csv(log_test_file, header=True, index=False)
    
    print("Saving net")
    torch.save(network.state_dict(), 'trained_models/cnn/%s.pth' % opt.model_name)

