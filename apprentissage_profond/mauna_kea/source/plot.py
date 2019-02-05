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
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--model_type', type=str, default='cnn',  help='type of model')
parser.add_argument('--model_name', type=str, default='MaunaNet4',  help='name of the model for log')
parser.add_argument('--criterion', type=str, default='cross_entropy',  help='name of the criterion to use')
parser.add_argument('--optimizer', type=str, default='adam',  help='name of the optimizer to use')
parser.add_argument('--lr', type=float, default=1e-3,  help='learning rate')
opt = parser.parse_args()

# =========================== LOAD FILES =========================== # 
log_train_file = "./log/%s/logs_train_%s.csv" % (opt.model_type, opt.model_name)
log_test_file = "./log/%s/logs_test_%s.csv" % (opt.model_type, opt.model_name)

if not os.path.exists(log_train_file): 
    raise ValueError("File %s does not exist" % log_train_file)
else:
    df_logs_train = pd.read_csv(log_train_file, header='infer')

if not os.path.exists(log_test_file):
    raise ValueError("File %s does not exist" % log_train_file)
else:
    df_logs_test = pd.read_csv(log_test_file, header='infer')

# ====================== PLOT LEARNING CURVES ====================== #
mask = (df_logs_train.model == opt.model_name) & (df_logs_train.crit == opt.criterion) \
        & (df_logs_train.optim == opt.optimizer) & (df_logs_train.lr == opt.lr)

graph_dir = "./graphs/%s" % opt.model_type
graph_file = os.path.join(graph_dir, "lc_%s_crit_%s_optim_%s_lr_%.2g.png" % (opt.model_name, opt.criterion, 
                                                                             opt.optimizer, opt.lr))
if not os.path.exists(graph_dir):
    os.mkdir(graph_dir)


# TRAIN/TEST

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 12))
ax.plot(np.arange(1, opt.n_epoch+1), df_logs_train[mask].acc, color="steelblue", label="train")
ax.plot(np.arange(1, opt.n_epoch+1), df_logs_test[mask].acc, color="limegreen", label="test")
title = "LC %s - %s, %s, lr %.2g" % (opt.model_name, opt.criterion, opt.optimizer, opt.lr)
ax.set_title(title, fontsize=25, fontweight="bold")
ax.set_xlabel("epoch", fontsize=20)
ax.set_ylabel("acc", fontsize=20)
ax.legend(loc="best", fontsize=20)
ax.grid(True, ls="--", lw=1)
fig.savefig(graph_file, format="png")
plt.close("all")

# DETAIL PER CATEGORY

cmap = cm .get_cmap("tab10")
graph_file = os.path.join(graph_dir, "lc_cat_%s_crit_%s_optim_%s_lr_%.2g.png" % (opt.model_name, opt.criterion, 
                                                                                 opt.optimizer, opt.lr))

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
title = "LC %s - %s, %s, lr %.2g" % (opt.model_name, opt.criterion, opt.optimizer, opt.lr)
ax.set_title(title, fontsize=25, fontweight="bold")
ax.set_xlabel("epoch", fontsize=20)
ax.set_ylabel("acc", fontsize=20)
ax.legend(loc="best", fontsize=20)
ax.grid(True, ls="--", lw=1)
fig.savefig(graph_file, format="png")
plt.close("all")

