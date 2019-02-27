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
sys.path.append("./source")
from auxiliary.dataset import *
from auxiliary.utils import *
from models.models_cnn import *

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M") 

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--data_aug', type=int, default=0 , help='1 or more for data augmentation')
parser.add_argument('--n_classes', type=int, default=4, help='number of classes')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--st_epoch', type=int, default=0, help='if continuing training, epoch from which to continue')
parser.add_argument('--model_type', type=str, default='pretrained_test',  help='type of model')
parser.add_argument('--model_name', type=str, default='AlexNet',  help='name of the model for log')
parser.add_argument('--fz_depth', type=int, default=10,  help='depth of freezed layers')
parser.add_argument('--model', type=str, default=None,  help='optional reload model path')
parser.add_argument('--criterion', type=str, default='cross_entropy',  help='name of the criterion to use')
parser.add_argument('--optimizer', type=str, default='sgd',  help='name of the optimizer to use')
parser.add_argument('--lr', type=float, default=1e-3,  help='learning rate')
parser.add_argument('--cuda', type=int, default=0, help='set to 1 to use cuda')
parser.add_argument('--random_state', type=int, default=0, help='random state for the split of data')
opt = parser.parse_args()

# ========================== TRAINING AND TEST DATA ========================== #
dataset_train = MaunaKea(train=True, data_aug=opt.data_aug, rgb=True, random_state=opt.random_state)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

dataset_test = MaunaKea(train=False, data_aug=opt.data_aug, rgb=True, random_state=opt.random_state)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

print('training set size %d' % len(dataset_train))
print('test set size %d' % len(dataset_test))

n_batch = len(dataset_train)//opt.batch_size + 1

# ========================== NETWORK AND OPTIMIZER ========================== #
network = eval("%s(n_classes=opt.n_classes, fz_depth=opt.fz_depth)" % opt.model_name)

if opt.model is not None:
    network.load_state_dict(torch.load(opt.model))
    print("Weights from %s loaded" % opt.model)

#elif opt.model_name == "densenet121":
#    # Let weights of first 2 layers unchanged
#    for param in list(model.children())[0][0].parameters():
#        param.requires_grad = False
#    for param in list(model.children())[0][1].parameters():
#        param.requires_grad = False
#
#    # _Dense Block is composed of 6 dense layers
#    # Let weights of first 3 layers unchanged
#    for i in range(3):
#        for param in list(model.children())[0][4][i].parameters():
#            param.requires_grad = False
#
#elif opt.model_name == "inception_v3":
#    # First 7 layers of inception_v3 are BasicConv2d and MaxPool2D
#    # Next 4 are InceptionA layers
#    for i in range(11):
#        for param in list(model.children())[0][i].parameters():
#            param.requires_grad = False
#
#elif opt.model_name == "se_resnet50":
#    # CUDA memory is enough to update all parameters
#    pass
#
#elif opt.model_name == "inceptionresnetv2":
#    # First 15 layers, only update params of last one
#    for i in range(14):
#        for param in list(model.children())[0][i].parameters():
#            param.requires_grad = False

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
log_path = os.path.join('log', opt.model_type)
if not os.path.exists(log_path):
    os.mkdir(log_path)

save_path = os.path.join('trained_models', opt.model_type)
if not os.path.exists(save_path):
    os.mkdir(save_path)

log_file = os.path.join(log_path, 'pretrained_%s_fz_%d.txt' % (opt.model_name, opt.fz_depth))
if not os.path.exists(log_file):
    with open(log_file, 'a') as log:
        log.write(str(network) + '\n')
        for nc, child in list(network.named_children()): 
            for nl, layer in list(child.named_children()):  
                for param in layer.parameters(): 
                    if param.requires_grad:
                        log.write("Child %s layer %s param is not frozen\n" % (nc, nl))
                    else: 
                        log.write("Child %s layer %s param is frozen\n" % (nc, nl))
        log.write("train patients %s\n" % dataset_train._train_pat)
        log.write("train labels %s\n" % np.bincount([x[1] for x in dataset_train._data]))
        log.write("test patients %s\n" % dataset_test._test_pat)
        log.write("test labels %s\n\n" % np.bincount([x[1] for x in dataset_test._data]))

log_train_file = "./log/%s/logs_train_%s_fz_%d.csv" % (opt.model_type, opt.model_name, opt.fz_depth)
log_test_file = "./log/%s/logs_test_%s_fz_%d.csv" % (opt.model_type, opt.model_name, opt.fz_depth)

if not os.path.exists(log_train_file): 
    df_logs_train = pd.DataFrame(columns=['model', 'fz_depth', 'data_aug', 'random_state', 'epoch', 'n_epoch', 'date', 
                                          'loss', 'acc', 'lr', 'optim', 'crit',
                                          'pred_0_0', 'pred_0_1', 'pred_0_2', 'pred_0_3',
                                          'pred_1_0', 'pred_1_1', 'pred_1_2', 'pred_1_3', 
                                          'pred_2_0', 'pred_2_1', 'pred_2_2', 'pred_2_3', 
                                          'pred_3_0', 'pred_3_1', 'pred_3_2', 'pred_3_3'])
else:
    df_logs_train = pd.read_csv(log_train_file, header='infer')

if not os.path.exists(log_test_file):
    df_logs_test = pd.DataFrame(columns=['model', 'fz_depth', 'data_aug', 'random_state', 'epoch', 'n_epoch', 'date', 
                                         'loss', 'acc', 'lr', 'optim', 'crit',
                                         'pred_0_0', 'pred_0_1', 'pred_0_2', 'pred_0_3',
                                         'pred_1_0', 'pred_1_1', 'pred_1_2', 'pred_1_3', 
                                         'pred_2_0', 'pred_2_1', 'pred_2_2', 'pred_2_3', 
                                         'pred_3_0', 'pred_3_1', 'pred_3_2', 'pred_3_3'])
else:
    df_logs_test = pd.read_csv(log_test_file, header='infer')

value_meter_train = AccuracyValueMeter(opt.n_classes)
value_meter_test = AccuracyValueMeter(opt.n_classes)

# ====================== LEARNING LOOP ====================== #
for epoch in range(opt.st_epoch, opt.n_epoch):
    # TRAINING
    st_time = time.time()
    network.train()
    value_meter_train.reset()
    loss_train = 0
    
    for batch, (fn, label, data) in enumerate(loader_train):
        if opt.cuda:
            label, data = label.cuda(), data.cuda()
    
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

    loss_train = float(loss_train.cpu())/n_batch
    dt = time.time()-st_time
    s_time = "%d min %d sec" % (dt//60, dt%60)
    
    print('='*40)
    print('[train epoch %d/%d] | loss %.3g | nw acc %.3g | time %s' % (epoch+1, opt.n_epoch, loss_train, value_meter_train.acc, s_time))
    for i in range(opt.n_classes):
        print('cat %d: %s and %s' % (i, value_meter_train.sum[i], [float("{0:0.4f}".format(f)) for f in value_meter_train.avg[i]]))
    print('='*40)
    
    with open(log_file, 'a') as log:
        log.write('[train epoch %d/%d] | loss %.5g | nw acc %.3g | time %s' % (epoch+1, opt.n_epoch, loss_train, value_meter_train.acc, s_time) +  '\n')
        for i in range(opt.n_classes):
            log.write('cat %d: %s and %s' % (i, value_meter_train.sum[i], [float("{0:0.4f}".format(f)) for f in value_meter_train.avg[i]]) + '\n')

    row_train = {'model': opt.model_name, 
                 'fz_depth': opt.fz_depth,
                 'data_aug': opt.data_aug,
                 'random_state': opt.random_state,
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
            row_train["pred_%d_%d" % (i, j)] = value_meter_train.sum[i][j]
    
    df_logs_train = df_logs_train.append(row_train, ignore_index=True)
    df_logs_train.to_csv(log_train_file, header=True, index=False)

    # TEST
    st_time = time.time()
    network.eval()
    value_meter_test.reset()
    loss_test = 0
    
    with torch.no_grad():
        if opt.data_aug:
            df = None
            for i in range(opt.data_aug):
                df_epoch = pd.DataFrame({'fn': [], 'label': [], 'pred_%d' % i: []})
                for batch, (fn, label, data) in enumerate(loader_test):
                    if opt.cuda:
                        label, data = label.cuda(), data.cuda()
                    
                    output = network(data)
                    loss = criterion(output, label)
                    loss_test += loss
           
                    pred = output.cpu().data.numpy().argmax(axis=1)
                    label = label.cpu().data.numpy()
                    fn = np.array(fn)
                    
                    df_epoch = pd.concat((df_epoch, pd.DataFrame({'fn': fn, 'label':label, 'pred_%d' % i: pred})), axis=0)
           
                if df is None:
                    df = df_epoch
                else:
                    df = df.merge(df_epoch, on=['fn', 'label'], how='left')
                print("Pass [%d/%d] over test data over" % (i+1, opt.data_aug))
                      
            print(df.head())
            # Predictions are the most frequently predicted label across the different crops
            pred = df.loc[:, [x for x in df.columns if 'pred' in x]].mode(axis=1).values[:, 0].astype(int)
            label = df.label.values.astype(int)
            value_meter_test.update(pred, label, len(dataset_test))
            loss_test = float(loss_test.cpu())/(opt.data_aug*n_batch)
        else:
            for batch, (fn, label, data) in enumerate(loader_test):
                if opt.cuda:
                    label, data = label.cuda(), data.cuda()
                
                output = network(data)
                loss = criterion(output, label)
                loss_test += loss
        
                pred = output.cpu().data.numpy().argmax(axis=1)
                label = label.cpu().data.numpy()
                value_meter_test.update(pred, label, opt.batch_size)
            loss_test = float(loss_test.cpu())/n_batch
    
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
                'fz_depth': opt.fz_depth,
                'data_aug': opt.data_aug,
                'random_state': opt.random_state,
                'epoch': epoch+1, 
                'n_epoch': opt.n_epoch, 
                'date': get_time(), 
                'loss': loss_test, 
                'acc': value_meter_test.acc,
                'lr': opt.lr,
                'optim': opt.optimizer,
                'crit': opt.criterion}
    
    for i in range(opt.n_classes):
        for j in range(opt.n_classes):
            row_test["pred_%d_%d" % (i, j)] = value_meter_test.sum[i][j]
    
    df_logs_test = df_logs_test.append(row_test, ignore_index=True)
    df_logs_test.to_csv(log_test_file, header=True, index=False)
    
    print("Saving net")
    torch.save(network.state_dict(), os.path.join(save_path, '%s_fz_%d.pth' % (opt.model_name, opt.fz_depth)))

# python source/train/train_pretrained.py --data_aug 0 --n_epoch 100 --model_type pretrained_2 --model_name AlexNet --fz_depth 8 --lr 0.01 --cuda 1 --random_state 10
