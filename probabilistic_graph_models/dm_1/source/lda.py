# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:00:26 2018
Implements a generative model

@author: Yoann Pradat
"""
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

class LDA(object):
    """
    We assume that y has Bernoulli distribution with parameter pi and 
    the conditional distribution of x|y=i is Gaussian with parameters
    mu_i and Sigma.
    
    The following class implements MLE of these parameters and also predicts
    p(y=1|x).
    """
    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y):
        n = X.shape[0]
        
        self.pi = sum(y == 1)/n
        self.mu_0 = (X*np.where(y == 0, 1, 0).reshape(-1, 1)).sum(axis=0)/sum(y == 0)
        self.mu_1 = (X*np.where(y == 1, 1, 0).reshape(-1, 1)).sum(axis=0)/sum(y == 1)
    
        self.mu_0 = self.mu_0.reshape(-1, 1)
        self.mu_1 = self.mu_1.reshape(-1, 1)
        self.sigma = np.zeros((2, 2));

        for i in range(n):
            if y[i] == 0:
                self.sigma += (X[i, :].reshape(-1, 1)-self.mu_0).dot((X[i, :].reshape(-1, 1)-self.mu_0).T)/n
            else:
                self.sigma += (X[i, :].reshape(-1, 1)-self.mu_1).dot((X[i, :].reshape(-1, 1)-self.mu_1).T)/n

        # Coefficients a and b to predict p(y=1|x)
        self.a = np.log(self.pi/(1-self.pi)) + \
        0.5*(self.mu_0.T.dot(np.linalg.inv(self.sigma)).dot(self.mu_0) - \
             self.mu_1.T.dot(np.linalg.inv(self.sigma)).dot(self.mu_1))

        self.b = np.linalg.inv(self.sigma).dot(self.mu_1-self.mu_0)
        self.is_fitted = True
        return self
        
    def input(self, X):
        return X.dot(self.b) + self.a

    def activation(self, X):
        return sigmoid(self.input(X)) 

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Please fit the model before making predictions")
        else:
            return np.where(self.input(X) >= 0, 1, 0)
