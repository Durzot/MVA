# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:03:12 2018
HMM model and EM algorithm.

@author: Yoann Pradat
"""

import numpy as np
import pandas as pd
import os
from importlib import reload

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

os.chdir("./source")
import hmm
reload(hmm)
from hmm import HMM
os.chdir("..")

df_train = pd.read_csv("classification_data_HWK2/EMGaussian.data", header=None, sep=" ", names=["x1","x2"])
df_test = pd.read_csv("classification_data_HWK2/EMGaussian.test", header=None, sep=" ", names=["x1","x2"])


"""
Functions to plot the results
"""

def plot_clusters(X, centers, labels, ax):
    k = centers.shape[0]
    colors = ('gold', 'royalblue', 'lightgreen', 'gray', 'turquoise')

    for center, l in zip(centers, range(k)):
        ax.scatter(X[labels==l, 0], X[labels==l,1], c=colors[l], marker='x', label="cluster %d"%l)
        ax.scatter(center[0], center[1], c="black", marker='o', s=60)

def plot_ellipses(centers, heights, widths, alpha, ax, angles=None):
    k = centers.shape[0]
    colors = ('gold', 'royalblue', 'lightgreen', 'gray', 'turquoise')

    if angles is None:
        angles = np.zeros(heights.shape[0])
    
    ells = []
    for center, height, width, angle in zip(centers, heights, widths, angles):
        ells.append(Ellipse(xy=center, width=width, height=height, angle=angle))

    for l,e in enumerate(ells):
        ax.add_artist(e)
        e.set_alpha(alpha)
        e.set_facecolor("salmon")

def extract_params_ellipse(Sigma):
    eigvals, eigvecs = np.linalg.eig(Sigma)

    eigpairs = [(np.abs(eigvals[i]), eigvecs[:,i]) for i in range(len(eigvals))]
    eigpairs.sort(key=lambda k:k[0], reverse=True)

    width = np.sqrt(eigpairs[0][0])
    height = np.sqrt(eigpairs[1][0])
    angle = -np.arctan(eigpairs[0][1][0]/eigpairs[0][1][1])*180/np.pi
    
    return width, height, angle

"""
HMM model with K=4 possible states and observations conditional to the chain are normally distributed.
"""

K=4
seed = 1995
max_iter = 100
log_interval =  10
verbose = 1

clf = HMM(K=K, max_iter=max_iter, verbose=verbose, log_interval=log_interval)
clf = clf.fit(df_train.values)


print("Log-likelihood at convergence on train %.3f" % clf.lc)
print("Log-likelihood on test %.3f" % clf._get_lc(df_test.values, clf.predict(df_test.values)))

fig, ax = plt.subplots(nrows=1, ncols=1)
plot_clusters(df_train.values, clf.mu, clf.labels, ax=ax)

widths = np.zeros(K)
heights = np.zeros(K)
angles = np.zeros(K)

for j in range(K):
    widths[j], heights[j], angles[j] = extract_params_ellipse(clf.Sigma[j])

plot_ellipses(clf.mu, 2*1.65*widths, 2*1.65*heights, alpha=0.35, ax=ax, angles=angles)

ax.legend(loc="upper left", fontsize=8)
ax.axis("equal")
ax.set_xlabel("x1", fontsize=10, fontweight="bold")
ax.set_ylabel("x2", fontsize=10, fontweight="bold")
ax.set_title("HMM model with K=4 possible states", fontsize=15, fontweight="bold")

plt.savefig("figures/hmm.png")


