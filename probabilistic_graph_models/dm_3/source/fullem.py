# -*- coding: utf-8 -*-
"""
Implements Expectation-Maximization algorithm with full covariance matrices.

@author: Yoann Pradat
"""

import numpy as np
from kmeans import KMeans

class FullGMM(object):
    """
    We observe xi for i=1,...,n and we want to infer a mixture of gaussians parametrized by the unobserved latent
    variable z.

    The vecteur theta of parameters is initialized using KMeans algorithm and then updated using EM algorithm for a
    mixture of Gaussians with full variance.
    """

    def __init__(self, k, max_iter=1000, tol=1e-6):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        n, d = X.shape

        # Parameters to be estimated
        self.pi = np.zeros(self.k)
        self.mu = np.zeros((self.k,d))
        self.Sigma = np.array([np.identity(d)]*self.k)
        tau = np.zeros((n ,self.k))

        # Label for each observation, i.e distribution to which it belongs
        self.labels = np.zeros(n)
            
        # Initialize the parameters using K-means
        # The idea is to use K-means to do this initialization
        self.kmeans = KMeans(k=self.k, order=2)
        self.kmeans.fit(X)

        for j in range(self.k):
            self.pi[j] = sum(self.kmeans.labels==j)/n

        self.mu = self.kmeans.centers
        self.labels = self.kmeans.labels
        for j in range(self.k):
            tmp = X[self.kmeans.labels==j] - X[self.kmeans.labels==j].mean(axis=0)
            self.Sigma[j] = tmp.T.dot(tmp)/n

        # Initial complete log likelihood
        self.list_lc = [-np.inf]
        self.lc = self._get_lc(X, self.pi, self.mu, self.Sigma, self.labels)
        
        n_iter = 0
        while n_iter < self.max_iter and np.abs(self.lc-self.list_lc[-1]) > self.tol:
            tau = self._get_tau(X, self.pi, self.mu, self.Sigma)
            self.pi = self._update_pi(tau)
            self.mu = self._update_mu(X, tau)
            self.Sigma = self._update_Sigma(X, self.mu, tau)
            self.labels = np.argmax(tau, axis=1)
            self.list_lc.append(self.lc)
            self.lc = self._get_lc(X, self.pi, self.mu, self.Sigma, self.labels)
            n_iter += 1

        self.list_lc = self.list_lc[1:]
        self.list_lc.append(self.lc)
        return self

    def _get_tau(self, X, pi, mu, Sigma):
        n,d = X.shape
        tau = np.zeros((n, self.k))
        for i in range(n):
            for j in range(self.k):
                tau[i,j] = pi[j]*np.linalg.det(Sigma[j])**(-1/2.)*np.exp(- 1./2.*(X[i]-mu[j]).dot(np.linalg.inv(Sigma[j])).dot(X[i]-mu[j]))
        tau = (tau.T/tau.T.sum(axis=0)).T
        return tau

    def _update_pi(self, tau):
        return tau.mean(axis=0)

    def _update_mu(self, X, tau):
        n, d = X.shape
        mu = np.zeros((self.k,d))
        for j in range(self.k):
            mu[j] = np.average(X, axis=0, weights=tau[:, j])
        return mu

    def _update_Sigma(self, X, mu, tau):
        n, d = X.shape
        Sigma = np.array([np.identity(d)]*self.k)
        for j in range(self.k):
            for i in range(n):
                Sigma[j] += (X[i]-mu[j]).reshape(-1, 1).dot((X[i]-mu[j]).reshape(1, -1))*tau[i, j]
            Sigma[j] /= tau[:, j].sum()
        return Sigma

    def _get_lc(self, X, pi, mu, Sigma, labels):
        n, d = X.shape
        k = pi.shape[0]
        lc = 0
        for i in range(n):
            for j in range(k):
                if labels[i] == j:
                    lc += np.log(pi[j]) - d/2.*np.log(2*np.pi) - 1./2.*np.log(np.linalg.det(Sigma[j])) - 1./2.*(X[i]-mu[j]).dot(np.linalg.inv(Sigma[j])).dot(X[i]-mu[j])
        return lc

    def predict(self, X):
        tau = self._get_tau(X, self.pi, self.mu, self.Sigma)
        return np.argmax(tau, axis=1)

    def get_lc(self, X):
        tau = self._get_tau(X, self.pi, self.mu, self.Sigma)
        labels = np.argmax(tau, axis=1)
        return self._get_lc(X, self.pi, self.mu, self.Sigma, labels)

 

