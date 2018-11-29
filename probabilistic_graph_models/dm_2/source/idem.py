# -*- coding: utf-8 -*-
"""
Implements Expectation-Maximization algorithm with covariance matrices proportional to identity.

@author: Yoann Pradat
"""

import numpy as np
from kmeans import KMeans

class IdGMM(object):
    """
    We observe xi for i=1,...,n and we want to infer a mixture of gaussians parametrized by the unobserved latent
    variable z.

    The vecteur theta of parameters is initialized using KMeans algorithm and then updated using EM algorithm for a
    mixture of Gaussians with spherical variance.
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
        self.sigma2 = np.zeros(self.k)
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
            self.sigma2[j] = X[self.kmeans.labels==j].var(axis=0).mean()

        # Initial complete log likelihood
        self.list_lc = [-np.inf]
        self.lc = self._get_lc(X, self.pi, self.mu, self.sigma2, self.labels)
        
        n_iter = 0
        while n_iter < self.max_iter and np.abs(self.lc-self.list_lc[-1]) > self.tol:
            tau = self._get_tau(X, self.pi, self.mu, self.sigma2)
            self.pi = self._update_pi(tau)
            self.mu = self._update_mu(X, tau)
            self.sigma2 = self._update_sigma2(X, self.mu, tau)
            self.labels = np.argmax(tau, axis=1)
            self.list_lc.append(self.lc)
            self.lc = self._get_lc(X, self.pi, self.mu, self.sigma2, self.labels)
            n_iter += 1

        self.list_lc = self.list_lc[1:]
        self.list_lc.append(self.lc)
        return self

    def _get_tau(self, X, pi, mu, sigma2):
        n,d = X.shape
        tau = np.zeros((n, self.k))
        for i in range(n):
            for j in range(self.k):
                tau[i,j] = pi[j]*sigma2[j]**(-d/2.)*np.exp(- 1./(2.*sigma2[j])*(X[i]-mu[j]).dot(X[i]-mu[j]))
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

    def _update_sigma2(self, X, mu, tau):
        n, d = X.shape
        sigma2 = np.zeros(self.k)
        for j in range(self.k):
            for i in range(n):
                sigma2[j] += (X[i]-mu[j]).dot(X[i]-mu[j])*tau[i, j]
            sigma2[j] /= d*tau[:, j].sum()
        return sigma2

    def _get_lc(self, X, pi, mu, sigma2, labels):
        n, d = X.shape
        k = pi.shape[0]
        lc = 0
        for i in range(n):
            for j in range(k):
                if labels[i] == j:
                    lc += np.log(pi[j])  - d/2.*np.log(2*np.pi*sigma2[j]) - 1./(2.*sigma2[j])*(X[i]-mu[j]).dot(X[i]-mu[j])
        return lc

    def predict(self, X):
        tau = self._get_tau(X, self.pi, self.mu, self.sigma2)
        return np.argmax(tau, axis=1)

    def get_lc(self, X):
        tau = self._get_tau(X, self.pi, self.mu, self.sigma2)
        labels = np.argmax(tau, axis=1)
        return self._get_lc(X, self.pi, self.mu, self.sigma2, labels)

    
