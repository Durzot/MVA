# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 23:35 2018
Implements a generative model

@author: Yoann Pradat
"""
import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

class QDA(object):
    """
    We assume that y has Bernoulli distribution with parameter pi and 
    the conditional distribution of x|y=i is Gaussian with parameters
    mu_i and Sigma_i. (Note the dependence of Sigma on i)
    
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
        self.sigma_0 = np.zeros((2, 2));
        self.sigma_1 = np.zeros((2, 2));

        for i in range(n):
            if y[i] == 0:
                self.sigma_0 += (X[i, :].reshape(-1, 1)-self.mu_0).dot((X[i, :].reshape(-1, 1)-self.mu_0).T)/sum(y == 0)
            else:
                self.sigma_1 += (X[i, :].reshape(-1, 1)-self.mu_1).dot((X[i, :].reshape(-1, 1)-self.mu_1).T)/sum(y == 1)

        # Coefficients a, b and c to predict p(y=1|x)
        sigma_0_inv = np.linalg.inv(self.sigma_0)
        sigma_1_inv = np.linalg.inv(self.sigma_1)

        self.a = np.log(self.pi/(1-self.pi)) + \
        0.5*(self.mu_0.T.dot(sigma_0_inv).dot(self.mu_0) - \
             self.mu_1.T.dot(sigma_1_inv).dot(self.mu_1)) + \
        0.5*np.log(np.linalg.det(self.sigma_1)/np.linalg.det(self.sigma_0))

        self.b = sigma_1_inv.dot(self.mu_1) - sigma_0_inv.dot(self.mu_0)
        self.c = 0.5*(sigma_0_inv - sigma_1_inv)
        self.is_fitted = True

        return self
        
    def input(self, X):
        return np.diag(X.dot(self.c).dot(X.T)).reshape(-1, 1) + X.dot(self.b) + self.a

    def activation(self, X):
        return sigmoid(self.input(X)) 

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Please fit the model before making predictions")
        else:
            return np.where(self.input(X) >= 0, 1, 0)

