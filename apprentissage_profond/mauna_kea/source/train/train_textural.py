# -*- coding: utf-8 -*-
"""
Classifiers on merged high-level features (from CNN) and traditional texture and color features.
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
import warnings

import torch
sys.path.append("./source")
from auxiliary.dataset import *
from auxiliary.utils import *
from models.models_cnn import *

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M") 

# =========================== PARAMETERS =========================== # 
parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=4, help='number of classes')
parser.add_argument('--model_type', type=str, default='cnn_Mauna_decay_20_20',  help='type of model')
parser.add_argument('--model_name', type=str, default='MaunaNet2',  help='name of the model for log')
parser.add_argument('--optimizer', type=str, default="sgd",  help='optimizer used to train the model')
parser.add_argument('--n_splits', type=int, default=5, help='number of splits for cv of params')
parser.add_argument('--n_jobs', type=int, default=3, help='number of jobs for the gridsearch')
parser.add_argument('--verbose', type=int, default=1, help='verbose for the gridsearch')
parser.add_argument('--random_state', type=int, default=50, help='random state for the split of data')
opt = parser.parse_args()

data_path = './data/TransformedData/%s/%s_%s' % (opt.model_type, opt.model_name, opt.optimizer)

if not os.path.exists(data_path):
    raise ValueError("There is no path %s" % data_path)

# ========================== MERGE THE FEATURES =========================== #
# High level features
data_hl_feat_train = pd.read_csv(os.path.join(data_path, "data_hl_feat_train.csv"), header='infer')
data_hl_feat_test = pd.read_csv(os.path.join(data_path, "data_hl_feat_test.csv"), header='infer')

# Traditional features
data_textural_train = pd.read_csv('./data/TransformedData/data_textural_train.csv', header='infer')

data_merge_train = data_hl_feat_train.merge(data_textural_train, how='left', on=['image_filename', 'class_number'])
data_merge_test = data_hl_feat_test.merge(data_textural_train, how='left', on=['image_filename', 'class_number'])

# Train set
info_cols =  ['image_filename', 'class_number', 'patient']
text_cols = [x for x in data_merge_train.columns if 'hf_' not in x and x not in info_cols] 
hl_cols = [x for x in data_merge_train.columns if 'hf_' in x] 
merge_cols = [x for x in data_merge_train.columns if x not in info_cols]

X_text_train = data_merge_train.loc[:, text_cols].values
X_hl_train = data_merge_train.loc[:, hl_cols].values
X_merge_train = data_merge_train.loc[:, merge_cols].values
y_train = data_merge_train.loc[:, 'class_number'].values

# Test set (internal)
X_text_test = data_merge_test.loc[:, text_cols].values
X_hl_test = data_merge_test.loc[:, hl_cols].values
X_merge_test = data_merge_test.loc[:, merge_cols].values
y_test = data_merge_test.loc[:, 'class_number'].values

# =========================== DATA SPLIT ============================ #
# Build train/test splits at patient level
kf = KFold(n_splits=opt.n_splits, shuffle=True, random_state=0)
cv_splits = []
for train_pat, test_pat in kf.split(data_merge_train.patient.unique()):
    train_idx = data_merge_train.loc[data_merge_train.patient.isin(train_pat)].index
    test_idx = data_merge_train.loc[data_merge_train.patient.isin(test_pat)].index
    cv_splits.append((train_idx, test_idx))

###########################################################################
# ========================== CLASSIFY THE DATA ========================== #
###########################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

log_train_file = "./log/%s/logs_clf_train_%s_%s.csv" % (opt.model_type, opt.model_name, opt.optimizer)
log_test_file = "./log/%s/logs_clf_test_%s_%s.csv" % (opt.model_type, opt.model_name, opt.optimizer)

df_logs_train = pd.DataFrame(columns=['model', 'clf', 'features', 'params', 'random_state', 'date', 'acc',
                                      'pred_0_0', 'pred_0_1', 'pred_0_2', 'pred_0_3',
                                      'pred_1_0', 'pred_1_1', 'pred_1_2', 'pred_1_3', 
                                      'pred_2_0', 'pred_2_1', 'pred_2_2', 'pred_2_3', 
                                      'pred_3_0', 'pred_3_1', 'pred_3_2', 'pred_3_3'])

df_logs_test = pd.DataFrame(columns=['model', 'clf', 'features', 'params', 'random_state', 'date', 'acc',
                                     'pred_0_0', 'pred_0_1', 'pred_0_2', 'pred_0_3',
                                     'pred_1_0', 'pred_1_1', 'pred_1_2', 'pred_1_3', 
                                     'pred_2_0', 'pred_2_1', 'pred_2_2', 'pred_2_3', 
                                     'pred_3_0', 'pred_3_1', 'pred_3_2', 'pred_3_3'])

value_meter_train = AccuracyValueMeter(opt.n_classes)
value_meter_test = AccuracyValueMeter(opt.n_classes)

for features, X_train, X_test in zip(['textural', 'high_level', 'merge'], 
                                     [X_text_train, X_hl_train, X_merge_train], 
                                     [X_text_test, X_hl_test, X_merge_test]):

    print("FEATURES %s" % features)
    
    # ========================== LOGISTIC REGRESSION ========================== #
    print("\n")
    print("="*40)
    print("Classifying with logistic regression...")
    print("="*40)
    print("\n")
    
    # Reset the meters for accuracy across each cat
    value_meter_train.reset()
    value_meter_test.reset()
    
    clf_name = 'logistic regression'
    clf = LogisticRegression(fit_intercept=True,
                             penalty='l2',
                             max_iter=300,
                             tol=1e-1,
                             multi_class='ovr',
                             solver='lbfgs')
    
    param_grid = {'C' : np.logspace(-2, 2, num=20)}
    gs = GridSearchCV(estimator=clf,
                      param_grid=param_grid,
                      scoring='accuracy',
                      n_jobs=opt.n_jobs,
                      verbose=opt.verbose,
                      return_train_score=True,
                      cv=cv_splits)
    
    t1 = time.time()
    gs.fit(X_train, y_train)
    t2 = time.time()
    print("Fitting time %.3g s" % (t2-t1))
    print("BEST PARAMETERS from gridsearch logistic regression %s" % gs.best_params_) 
    
    for param, train_score, test_score in zip(gs.cv_results_['params'], gs.cv_results_['mean_train_score'],
                                              gs.cv_results_['mean_test_score']):
        param_print = {k:round(v,3) if isinstance(v,float) else v for k,v in param.items()}
        print("For params %s mean train score %.3g and mean test score %.3g" % (param_print, train_score, test_score))
    
    y_pred_train = gs.predict(X_train)
    y_pred_test = gs.predict(X_test)
    print("Precision on the train set %.3f" % np.mean(y_pred_train==y_train))
    print("Precision on the test set %.3f" % np.mean(y_pred_test==y_test))
    
    value_meter_train.update(y_pred_train, y_train, y_train.shape[0])
    value_meter_test.update(y_pred_test, y_test, y_test.shape[0])
    best_param_print = {k:round(v,3) if isinstance(v,float) else v for k,v in gs.best_params_.items()}
    
    row_train = {'model': opt.model_name, 
                 'clf': clf_name,
                 'features': features,
                 'params': best_param_print,
                 'random_state': opt.random_state,
                 'date': get_time(), 
                 'acc': value_meter_train.acc}
    
    for i in range(opt.n_classes):
        for j in range(opt.n_classes):
            row_train["pred_%d_%d" % (i, j)] = value_meter_train.sum[i][j]
    
    df_logs_train = df_logs_train.append(row_train, ignore_index=True)
    df_logs_train.to_csv(log_train_file, header=True, index=False)                
    
    row_test = {'model': opt.model_name, 
                'clf': clf_name,
                'features': features,
                'params': best_param_print,
                'random_state': opt.random_state,
                'date': get_time(), 
                'acc': value_meter_test.acc}
    
    for i in range(opt.n_classes):
        for j in range(opt.n_classes):
            row_test["pred_%d_%d" % (i, j)] = value_meter_test.sum[i][j]
    
    df_logs_test = df_logs_test.append(row_test, ignore_index=True)
    df_logs_test.to_csv(log_test_file, header=True, index=False)        
    
    # ========================== RANDOM FOREST ========================== #
    print("\n")
    print("="*40)
    print("Classifying with random forest...")
    print("="*40)
    print("\n")
    
    # Reset the meters for accuracy across each cat
    value_meter_train.reset()
    value_meter_test.reset()
    
    clf_name = 'random forest'
    clf = RandomForestClassifier(random_state=opt.random_state,
                                 max_features='sqrt')
    
    param_grid = {'n_estimators' : [4, 5, 8, 10, 20, 30],
                  'max_depth': [2, 3, 4, 5, 6, 7, 8]}
    gs = GridSearchCV(estimator=clf,
                      param_grid=param_grid,
                      scoring='accuracy',
                      n_jobs=opt.n_jobs,
                      verbose=opt.verbose,
                      return_train_score=True,
                      cv=cv_splits)
    
    t1 = time.time()
    gs.fit(X_train, y_train)
    t2 = time.time()
    print("Fitting time %.3g s" % (t2-t1))
    print("BEST PARAMETERS from gridsearch random forest %s" % gs.best_params_) 
    
    for param, train_score, test_score in zip(gs.cv_results_['params'], gs.cv_results_['mean_train_score'],
                                              gs.cv_results_['mean_test_score']):
        param_print = {k:round(v,3) if isinstance(v,float) else v for k,v in param.items()}
        print("For params %s mean train score %.3g and mean test score %.3g" % (param_print, train_score, test_score))
    
    y_pred_train = gs.predict(X_train)
    y_pred_test = gs.predict(X_test)
    print("Precision on the train set %.3f" % np.mean(y_pred_train==y_train))
    print("Precision on the test set %.3f" % np.mean(y_pred_test==y_test))
    
    value_meter_train.update(y_pred_train, y_train, y_train.shape[0])
    value_meter_test.update(y_pred_test, y_test, y_test.shape[0])
    best_param_print = {k:round(v,3) if isinstance(v,float) else v for k,v in gs.best_params_.items()}
    
    row_train = {'model': opt.model_name, 
                 'clf': clf_name,
                 'features': features,
                 'params': best_param_print,
                 'random_state': opt.random_state,
                 'date': get_time(), 
                 'acc': value_meter_train.acc}
    
    for i in range(opt.n_classes):
        for j in range(opt.n_classes):
            row_train["pred_%d_%d" % (i, j)] = value_meter_train.sum[i][j]
    
    df_logs_train = df_logs_train.append(row_train, ignore_index=True)
    df_logs_train.to_csv(log_train_file, header=True, index=False)                
    
    row_test = {'model': opt.model_name, 
                'clf': clf_name,
                 'features': features,
                'params': best_param_print,
                'random_state': opt.random_state,
                'date': get_time(), 
                'acc': value_meter_test.acc}
    
    for i in range(opt.n_classes):
        for j in range(opt.n_classes):
            row_test["pred_%d_%d" % (i, j)] = value_meter_test.sum[i][j]
    
    df_logs_test = df_logs_test.append(row_test, ignore_index=True)
    df_logs_test.to_csv(log_test_file, header=True, index=False)        

