# -*- coding: utf-8 -*-
"""
Scrit for visualizing details of training i.e training cruves
Python 3 virtual environment 3.7_pytorch_sk

@author: Yoann Pradat
"""

import os
import argparse
import numpy as np 
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=4, help='number of classes')
parser.add_argument('--n_epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--model_type', type=str, default='pretrained_ResNet18_fz',  help='type of model')
parser.add_argument('--model_name', type=str, default='ResNet18',  help='name of the model for log')
parser.add_argument('--fz_depth', type=int, default=10,  help='depth of frozen')
parser.add_argument('--criterion', type=str, default='cross_entropy',  help='name of the criterion to use')
parser.add_argument('--optimizer', type=str, default='sgd',  help='name of the optimizer to use')
parser.add_argument('--lr', type=float, default=5e-3,  help='learning rate')
opt = parser.parse_args()

# =========================== LOAD FILES =========================== # 
log_train_file = "./log/%s/logs_train_%s_fz_%d.csv" % (opt.model_type, opt.model_name, opt.fz_depth)
log_test_file = "./log/%s/logs_test_%s_fz_%d.csv" % (opt.model_type, opt.model_name, opt.fz_depth)

if not os.path.exists(log_train_file): 
    raise ValueError("File %s does not exist" % log_train_file)
else:
    df_logs_train = pd.read_csv(log_train_file, header='infer')

if not os.path.exists(log_test_file):
    raise ValueError("File %s does not exist" % log_train_file)
else:
    df_logs_test = pd.read_csv(log_test_file, header='infer')

if df_logs_train.shape[0] > df_logs_test.shape[0]:
    df_logs_train = df_logs_train.drop(df_logs_train.index[-1])

# ====================== PLOT LEARNING CURVES ====================== #
mask = (df_logs_train.model == opt.model_name) & (df_logs_train.crit == opt.criterion) \
        & (df_logs_train.optim == opt.optimizer)

graph_dir = "./graphs/%s" % opt.model_type
graph_file = os.path.join(graph_dir, "lc_%s_crit_%s_optim_%s_lr_%.2g_fz_%d.png" % (opt.model_name, opt.criterion, 
                                                                                   opt.optimizer, opt.lr, opt.fz_depth))
if not os.path.exists(graph_dir):
    os.mkdir(graph_dir)


# TRAIN/TEST

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 12))
ax.plot(np.arange(1, opt.n_epoch+1), df_logs_train[mask].acc, color="steelblue", label="train")
ax.plot(np.arange(1, opt.n_epoch+1), df_logs_test[mask].acc, color="limegreen", label="test")
title = "LC %s_%d - %s, %s, lr %.2g" % (opt.model_name, opt.fz_depth, opt.criterion, opt.optimizer, opt.lr)
ax.set_title(title, fontsize=25, fontweight="bold")
ax.set_xlabel("epoch", fontsize=20)
ax.set_ylabel("acc", fontsize=20)
ax.legend(loc="best", fontsize=20)
ax.grid(True, ls="--", lw=1)
fig.savefig(graph_file, format="png")
plt.close("all")

# DETAIL PER CATEGORY

cmap = cm .get_cmap("tab10")
graph_file = os.path.join(graph_dir, "lc_cat_%s_crit_%s_optim_%s_lr_%.2g_fz_%d.png" % (opt.model_name, opt.criterion, 
                                                                                       opt.optimizer, opt.lr,
                                                                                       opt.fz_depth))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,12))
for i in range(opt.n_classes):
    s_sum = df_logs_train.loc[mask, [x for x in df_logs_train.columns if "pred_%d" % i in x]].sum(axis=1)
    s_pred = df_logs_train.loc[mask, "pred_%d_%d" % (i, i)]
    s_acc = s_pred.divide(s_sum, fill_value=0)
    ax.plot(np.arange(1, opt.n_epoch+1), s_acc, color=cmap(i), label="train cat %d" % i)

    s_sum = df_logs_test.loc[mask, [x for x in df_logs_test.columns if "pred_%d" % i in x]].sum(axis=1)
    s_pred = df_logs_test.loc[mask, "pred_%d_%d" % (i, i)]
    s_acc = s_pred.divide(s_sum, fill_value=0)
    ax.plot(np.arange(1, opt.n_epoch+1), s_acc, color=cmap(i), ls="--", marker='o', label="test cat %d" % i)
title = "LC %s_%d - %s, %s, lr %.2g" % (opt.model_name, opt.fz_depth, opt.criterion, opt.optimizer, opt.lr)
ax.set_title(title, fontsize=25, fontweight="bold")
ax.set_xlabel("epoch", fontsize=20)
ax.set_ylabel("acc", fontsize=20)
ax.legend(loc="best", fontsize=20)
ax.grid(True, ls="--", lw=1)
fig.savefig(graph_file, format="png")
plt.close("all")


#cmap = cm .get_cmap("tab10")
#colors = [cmap(i) for i in range(10)]
#
#graph_dir = "./graphs/pretrained_%s_fz" % opt.model_name
#graph_file = os.path.join(graph_dir, "lc_%s_optim_%s_lr_%.2g_all.png" % (opt.model_name, opt.optimizer, opt.lr))
#
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 12))
#for color, fz_depth in zip(colors, [3, 9, 15, 24, 30, 45]):
#    # Load log files
#    log_train_file = "./log/%s/logs_train_%s_fz_%d.csv" % (opt.model_type, opt.model_name, fz_depth)
#    log_test_file = "./log/%s/logs_test_%s_fz_%d.csv" % (opt.model_type, opt.model_name, fz_depth)
#    
#    df_logs_train = pd.read_csv(log_train_file, header='infer')
#    df_logs_test = pd.read_csv(log_test_file, header='infer')
#
#    if df_logs_train.shape[0] > df_logs_test.shape[0]:
#        df_logs_train = df_logs_train.drop(df_logs_train.index[-1])
#
#    ax.plot(np.arange(1, df_logs_train.shape[0]+1), df_logs_train.acc, lw=3, ls="-", color=color, label="train fz depth %d" % fz_depth)
#    ax.plot(np.arange(1, df_logs_test.shape[0]+1), df_logs_test.acc, lw=3, ls="--", color=color, label="test fz depth %d" % fz_depth)
#
#title = "Different freezing depths %s - %s, %s" % (opt.model_name, opt.criterion, opt.optimizer)
#ax.set_title(title, fontsize=25, fontweight="bold")
#ax.set_xlabel("epoch", fontsize=20)
#ax.set_ylabel("acc", fontsize=20)
#ax.legend(loc="best", fontsize=20)
#ax.grid(True, ls="--", lw=1)
#
#fig.savefig(graph_file, format="png")
#plt.close("all")


