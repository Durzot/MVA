# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:02 2018
Implements logistic regression using IRLS algorithm

@author: Yoann Pradat
"""
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

class LogisticRegression(object):
    """
    We assume that for a given observation X=x, the conditional distribution of y|X=x is a Bernoulli 
    with parameter theta = sigmoid(w^t*x + b).
    
    The following class implements conditional MLE of parameters w and b and also predicts
    p(y=1|x).

    """
    def __init__(self, max_iter=30, tol=0.1):
        self.is_fitted=False
        self.max_iter=max_iter
        self.tol=tol

    def fit(self, X, y):
        n = X.shape[0]
        X = np.hstack((X, np.ones((n, 1))))
        w = np.zeros(X.shape[1])

        # log likelihood previous and next for tolerance condition
        l_p = 0
        l_n = np.log(sigmoid(X.dot(w))).dot(y) + np.log(sigmoid(-X.dot(w))).dot(1-y)
        
        for _ in range(self.max_iter):
            l_p = l_n

            eta = sigmoid(X.dot(w))
            grad = X.T.dot(y-eta)
            hessian = -X.T.dot(np.diag(eta*(1-eta))).dot(X)
            w = w - np.linalg.inv(hessian).dot(grad)
            l_n = np.log(sigmoid(X.dot(w))).dot(y) + np.log(sigmoid(-X.dot(w))).dot(1-y)
           
            if np.abs(l_n-l_p) < self.tol:
                break

        self.w = w
        self.is_fitted = True
        return self

    def input(self, X):
        n = X.shape[0]
        X = np.hstack((X, np.ones((n,1))))
        return X.dot(self.w)

    def activation(self, X):
        return sigmoid(self.input(X))

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Please fit the model before making predictions")
        else:
            return np.where(self.input(X) >= 0, 1, 0)
