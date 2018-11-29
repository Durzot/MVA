# -*- coding: utf-8 -*-
"""
Created on Wed Nov 7 14:01:26 2018
Applies K-means and EM algorithms.

@author: Yoann Pradat
"""

import numpy as np
import pandas as pd
import os

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

os.chdir("./source")
from kmeans import KMeans
from idem import IdGMM
from fullem import FullGMM
os.chdir("..")

df_train = pd.read_csv("classification_data_HWK2/EMGaussian.data", header=None, sep=" ", names=["x1","x2"])
df_test = pd.read_csv("classification_data_HWK2/EMGaussian.test", header=None, sep=" ", names=["x1","x2"])

"""
KMeans algorithm on train data
"""

k=4
seeds = [123, 9121]
orders = [1,2]

def plot_clusters(X, centers, labels, ax):
    k = centers.shape[0]
    colors = ('gold', 'royalblue', 'lightgreen', 'gray', 'turquoise')

    for center, l in zip(centers, range(k)):
        ax.scatter(X[labels==l, 0], X[labels==l,1], c=colors[l], marker='x', label="cluster %d"%l)
        ax.scatter(center[0], center[1], c="black", marker='o', s=60)
        
fig, ax = plt.subplots(nrows=len(seeds), ncols=len(orders), figsize=(16,6))

for i, seed in enumerate(seeds):
    for j, order in enumerate(orders):
        clf = KMeans(k=k, order=order, seed=seed)
        clf = clf.fit(df_train.values)
        plot_clusters(df_train.values, clf.centers, clf.labels, ax=ax[i][j])
        ax[i][j].legend(loc="upper left", fontsize=8)
        ax[i][j].set_xlabel("x1", fontsize=10, fontweight="bold")
        ax[i][j].set_ylabel("x2", fontsize=10, fontweight="bold")
        ax[i][j].set_title("Distance order %d ; seed %d " % (order, seed), fontsize=15, fontweight="bold")
    
plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, left=0.05, right=0.95)
plt.savefig("figures/kmeans.png")

"""
EM algorithm with covariance matrices proportional to identity
"""

clf = IdGMM(k=k, max_iter=1000)
clf.fit(df_train.values)

print("Log-likelihood at convergence %.2f" % clf.get_lc(df_test.values))

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
                            

fig, ax = plt.subplots(nrows=1, ncols=1)

plot_clusters(df_train.values, clf.mu, clf.labels, ax=ax)
plot_ellipses(clf.mu, 2*1.65*np.sqrt(clf.sigma2), 2*1.65*np.sqrt(clf.sigma2), alpha=0.35, ax=ax)

ax.legend(loc="upper left", fontsize=8)
ax.axis("equal")
ax.set_xlabel("x1", fontsize=10, fontweight="bold")
ax.set_ylabel("x2", fontsize=10, fontweight="bold")
ax.set_title("EM algorithm covariances prop. to identity ", fontsize=15, fontweight="bold")

plt.savefig("figures/em_iso.png")

"""
EM algorithm with full covariance matrices
"""

clf = FullGMM(k=k, max_iter=1000)
clf.fit(df_train.values)

print("Log-likelihood at convergence %.2f" % clf.get_lc(df_test.values))

fig, ax = plt.subplots(nrows=1, ncols=1)
plot_clusters(df_train.values, clf.mu, clf.labels, ax=ax)

def extract_params_ellipse(Sigma):
    eigvals, eigvecs = np.linalg.eig(Sigma)

    eigpairs = [(np.abs(eigvals[i]), eigvecs[:,i]) for i in range(len(eigvals))]
    eigpairs.sort(key=lambda k:k[0], reverse=True)

    width = np.sqrt(eigpairs[0][0])
    height = np.sqrt(eigpairs[1][0])
    angle = -np.arctan(eigpairs[0][1][0]/eigpairs[0][1][1])*180/np.pi
    
    return width, height, angle

widths = np.zeros(k)
heights = np.zeros(k)
angles = np.zeros(k)

for j in range(k):
    widths[j], heights[j], angles[j] = extract_params_ellipse(clf.Sigma2[j])

plot_ellipses(clf.mu, 2*1.65*widths, 2*1.65*heights, alpha=0.35, ax=ax, angles=angles)

ax.legend(loc="upper left", fontsize=8)
ax.axis("equal")
ax.set_xlabel("x1", fontsize=10, fontweight="bold")
ax.set_ylabel("x2", fontsize=10, fontweight="bold")
ax.set_title("EM algorithm full covariances", fontsize=15, fontweight="bold")

plt.savefig("figures/em_full.png")

