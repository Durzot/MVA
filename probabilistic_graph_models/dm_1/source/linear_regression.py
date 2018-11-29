# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:00 2018
Implements linear regression using normal equation

@author: Yoann Pradat
"""
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

class LinearRegression(object):
    """
    We assume that for a given observation X=x, the conditional distribution of y|X=x is Gaussian with mean
    w^t*x+b and variance sigma^2.
    
    The following class solves the normal equation to estimate the parameters w and b and also predicts
    p(y=1|x).

    """
    def __init__(self):
        self.is_fitted=False

    def fit(self, X, y):
        n = X.shape[0]
        X = np.hstack((X, np.ones((n, 1))))
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
        self.w = w
        self.is_fitted = True
        return self

    def input(self, X):
        n = X.shape[0]
        X = np.hstack((X, np.ones((n,1))))
        return X.dot(self.w)
    
    def activation(self, X):
        return self.input(X)

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Please fit the model before making predictions")
        else:
            return np.where(self.activation(X) >= 0.5, 1, 0)
