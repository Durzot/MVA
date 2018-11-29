# -*- coding: utf-8 -*-
"""
Implements KMeans algorithm

@author: Yoann Pradat
"""
import numpy as np

def dist(x, y, p):
    return (sum(np.abs(x-y)**p))**(1./p)

class KMeans(object):
    """
    The algorithm tries to minimize the total distance of the points to the center of the cluster they belong to.
    The default distance is the euclidean (order=2), it can be changed by the user
    Centers are randomly initialized and then changed iteratively as the points cluster assignments change. 
    """

    def __init__(self, k, order=2, seed=0):
        self.k = k
        self.order = order
        self.seed = seed
        self.is_fitted = False

    def fit(self, X):
        n, d = X.shape

        self.centers = np.zeros((self.k, d))
        self.labels = np.zeros(n) 

        # Random initialization
        np.random.seed(self.seed)
        rand_perm = np.random.permutation(n)
        for l in range(self.k):
            self.centers[l] = X[rand_perm[l]]

        # Iterate until convergence ie no points change label         
        iterate = True
        while(iterate):
            iterate = False
            for i in range(n):
                label = np.argmin([dist(X[i],c,self.order) for c in self.centers])
                if self.labels[i] != label:
                    iterate = True
                self.labels[i] = label
            
            # Update centers
            for l in range(self.k):
                self.centers[l] = X[self.labels==l].mean(axis=0)
        
        self.is_fitted = True
        return self

    def predict(self, X):
        n, d = X.shape
        if self.is_fitted:
            labels = np.zeros(n)
            for i in range(n):
                labels[i] = np.armgin([dist(X[i],c,self.order) for c in self.centers])
            return labels
        else:
            raise ValueError("Please fit the model before making predictions")

    def fit_predict(self, X):
        self = self.fit(X)
        return self.labels
