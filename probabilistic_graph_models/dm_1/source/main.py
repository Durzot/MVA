# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:01:26 2018
Compares four different binary classification techniques on 2 datasets.

@author: Yoann Pradat
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

os.chdir("./source")
from lda import LDA
from logistic_regression import LogisticRegression
from linear_regression import LinearRegression
from qda import QDA
os.chdir("..")

df_A_train = pd.read_csv("classification_data_HWK1/classificationA.train", sep="\t", header=None, names=["x1","x2","y"])
df_A_test = pd.read_csv("classification_data_HWK1/classificationA.test", sep="\t", header=None, names=["x1","x2","y"])
df_B_train = pd.read_csv("classification_data_HWK1/classificationB.train", sep="\t", header=None, names=["x1","x2","y"])
df_B_test = pd.read_csv("classification_data_HWK1/classificationB.test", sep="\t", header=None, names=["x1","x2","y"])
df_C_train = pd.read_csv("classification_data_HWK1/classificationC.train", sep="\t", header=None, names=["x1","x2","y"])
df_C_test = pd.read_csv("classification_data_HWK1/classificationC.test", sep="\t", header=None, names=["x1","x2","y"])

def plot_decision_regions(X, y, classifier, ax, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    x2_min, x2_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
   
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y==cl,0],y=X[y==cl,1], alpha=0.8,
                   marker=markers[idx], c=colors[idx],
                   label=cl, edgecolor='black')
        
    if test_idx:
        X_test = X[test_idx, :]
        ax.scatter(X_test[:, 0], X_test[:, 1],
                   c='', edgecolor='black', alpha=1.0,
                   linewidth=0.5, marker='o',
                   s=10, label='test set')



X_A_train, y_A_train = df_A_train.iloc[:, :-1].values, df_A_train.iloc[:, -1].values
X_A_test, y_A_test = df_A_test.iloc[:, :-1].values, df_A_test.iloc[:, -1].values
X_B_train, y_B_train = df_B_train.iloc[:, :-1].values, df_B_train.iloc[:, -1].values
X_B_test, y_B_test = df_B_test.iloc[:, :-1].values, df_B_test.iloc[:, -1].values
X_C_train, y_C_train = df_C_train.iloc[:, :-1].values, df_C_train.iloc[:, -1].values
X_C_test, y_C_test = df_C_test.iloc[:, :-1].values, df_C_test.iloc[:, -1].values

for l in ["A", "B", "C"]:
    X_train = eval("X_%s_train" % l)
    y_train = eval("y_%s_train" % l)
    X_test = eval("X_%s_test" % l)
    y_test = eval("y_%s_test" % l)
   
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 18))
    ax = ax.flatten()

    print(l)
    
    """
    Run LDA
    """
    LDA_clf = LDA()
    LDA_clf.fit(X_train, y_train)
    
    lda_train_error = np.mean(LDA_clf.predict(X_train).flatten() != y_train)
    lda_test_error = np.mean(LDA_clf.predict(X_test).flatten() != y_test)
       
    plot_decision_regions(X=X_combined, y=y_combined, classifier=LDA_clf, 
                          test_idx=range(X_train.shape[0], X_train.shape[0]+X_test.shape[0]), ax=ax[0])
    ax[0].set_xlabel("x1", fontsize="large")
    ax[0].set_ylabel("x2", fontsize="large")
    ax[0].legend(loc="upper right", fontsize="large")
    ax[0].set_title("Generative model (LDA) on dataset %s" % l, fontsize="x-large", fontweight="bold")
    
    """
    Run logistic regression fitted by IRLS
    """
    logr_clf = LogisticRegression(max_iter=30, tol=0.01)
    logr_clf.fit(X_train, y_train)
    
    logr_train_error = np.mean(logr_clf.predict(X_train) != y_train)
    logr_test_error = np.mean(logr_clf.predict(X_test) != y_test)
    
    plot_decision_regions(X=X_combined, y=y_combined, classifier=logr_clf, 
                          test_idx=range(X_train.shape[0], X_train.shape[0]+X_test.shape[0]), ax=ax[1])
    ax[1].set_xlabel("x1", fontsize="large")
    ax[1].set_ylabel("x2", fontsize="large")
    ax[1].legend(loc="upper right", fontsize="large")
    ax[1].set_title("Logistic regression (IRLS) on dataset %s" % l, fontsize="x-large", fontweight="bold")
    
    """
    Run linear regression fitted by solving the normal equation
    """
    linr_clf = LinearRegression()
    linr_clf.fit(X_train, y_train)
    
    linr_train_error = np.mean(linr_clf.predict(X_train) != y_train)
    linr_test_error = np.mean(linr_clf.predict(X_test) != y_test)
    
    plot_decision_regions(X=X_combined, y=y_combined, classifier=linr_clf, 
                          test_idx=range(X_train.shape[0], X_A_train.shape[0]+X_test.shape[0]), ax=ax[2])
    ax[2].set_xlabel("x1", fontsize="large")
    ax[2].set_ylabel("x2", fontsize="large")
    ax[2].legend(loc="upper right", fontsize="large")
    ax[2].set_title("Linear regression (normal equation) on dataset %s" % l, fontsize="x-large", fontweight="bold")
   
    """
    Run QDA
    """
    QDA_clf = QDA()
    QDA_clf.fit(X_train, y_train)
    
    qda_train_error = np.mean(QDA_clf.predict(X_train).flatten() != y_train)
    qda_test_error = np.mean(QDA_clf.predict(X_test).flatten() != y_test)
       
    plot_decision_regions(X=X_combined, y=y_combined, classifier=QDA_clf, 
                          test_idx=range(X_train.shape[0], X_train.shape[0]+X_test.shape[0]), ax=ax[3])
    ax[3].set_xlabel("x1", fontsize="large")
    ax[3].set_ylabel("x2", fontsize="large")
    ax[3].legend(loc="upper right", fontsize="large")
    ax[3].set_title("Generative model (QDA) on dataset %s" % l, resolution=0.05, fontsize="x-large", fontweight="bold")

    plt.savefig("figures/dataset_%s.png" % l)
    
    """
    Make table of classification errors
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    train_errors = ["%.2g" % e for e in [lda_train_error, logr_train_error, linr_train_error, qda_train_error]]            
    test_errors = ["%.2g" % e for e in [lda_test_error, logr_test_error, linr_test_error, qda_test_error]]            
    cellText = [[tr,te] for tr, te  in zip(train_errors, test_errors)]

    colLabels=['train', 'test']
    rowLabels=['LDA', 'logr', 'linr', 'QDA']
    
    table = ax.table(cellText=cellText,
                     cellLoc='center',
                     rowLabels=rowLabels,
                     colLabels=colLabels,
                     loc='center',
                     bbox=[0.1, 0, 1.0, 1.0])
    
    table_props = table.properties()
    table_cells = table_props['child_artists']
    
    for i in range(1, 1+len(rowLabels)):
        table.get_celld()[(i, -1)].set_text_props(fontweight='bold')
    
    for j in range(0, len(colLabels)):
        table.get_celld()[(0, j)].set_text_props(fontweight='bold')

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    
    ax.axis('off')
    plt.savefig("figures/dataset_%s_table.png" % l)                       



