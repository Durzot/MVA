# -*- coding: utf-8 -*-
"""
Implements an HMM model where chain (qt) is indexed from 1 to T with K possible states and the observation ut
conditional to the state qt is normally distributed.

@author: Yoann Pradat
"""

import numpy as np
from fullem import FullGMM

class HMM(object):
    """
    We observe ut=(xt,yt) for t=1,...,T  and we want to compute conditional distribution p(qt|u1,...,uT) and p(qt,
    qt+1|u1,...,uT).

    The distributions will be computed using belief propagation algorithms. Paramaters to be learnt are initialized
    using a full covariance mixture of Gaussians model.

    Parameters
    ----------
    K: int
        Number of possible states for each node of the chain
    max_iter: int
        Maximum number of iterations in the EM algorithms for initializing parameters and learning parameters.
    tol: float
        Criteria on the absolute difference in  log-likelihoods to stop iterating.
    verbose: int
        If not 0, print messages during the learning.
    log_interval: int
        Controls frequency of messages in the iteration loop
    """

    # Computes efficiently log(e^x_1 + e^x_2 + ... + e^x_n)
    def _log_plus(self, x):
        i = np.argmax(x)
        return x[i] + np.log(np.exp(x-x[i]).sum())
    
    def __init__(self, K, max_iter=1000, tol=1e-6, verbose=0, log_interval = 10):
        self.K = K
        self.tol = tol
        self.max_iter = 1000
        self.verbose = verbose
        self.log_interval = log_interval

    def fit(self, U):
        T, d = U.shape

        # Parameters to be estimated
        self.pi = np.zeros(self.K)
        self.A = np.zeros((self.K, self.K))
        self.mu = np.zeros(self.K)
        self.Sigma = np.zeros((self.K, d, d))

        # Parameters for belief propagation
        self.log_alpha = np.zeros((T, self.K))
        self.log_beta = np.zeros((T, self.K))
        self.p_single = np.zeros((T, self.K))
        self.p_double = np.zeros((T-1, self.K, self.K))

        # Labels learnt from the parameters
        self.labels = np.zeros(T)

        # Initialize theta parameters for EM algorithm
        self._init_parameters(U)

        # Initial complete log likelihood
        self.list_lc = [-np.inf]
        self.lc = self._get_lc(U, self.labels)

        if self.verbose > 0:
            print("Initial log-likelihood with parameters from GMM and uniform transition matrix %.3f" % self.lc)
        
        n_iter = 0
        while n_iter < self.max_iter and np.abs(self.lc-self.list_lc[-1]) > self.tol:
            # Foward messages
            self._init_log_alpha(U)
            for t in range(1, T):
                self._forward_log_alpha(U, t)
    
            # Backward messagse
            self._init_log_beta(U)
            for t in reversed(range(T-1)):
                self._backward_log_beta(U, t)

            # Update conditional distribution after computing all messages (E step)
            self._update_p_single(U)
            self._update_p_double(U)

            # Update parameters (M step)
            self._update_Sigma(U)
            self._update_mu(U)
            self._update_A(U)
            self._update_pi(U)

            # Compute most probable state using Viberti algorithm
            self.labels = self._viberti(U)

            # Compute complete likelihood
            self.list_lc.append(self.lc)
            self.lc = self._get_lc(U, self.labels)
            n_iter += 1

            if self.verbose > 0 and n_iter % self.log_interval == 0:
                print("Log-likelihood at iteration %d/%d: %.3f ..." % (n_iter, self.max_iter, self.lc))

        self.list_lc = self.list_lc[1:]
        self.list_lc.append(self.lc)
        return self
                
    def _update_Sigma(self, U):
        T, d = U.shape
        for k in range(self.K):
            num = np.zeros((d,d))
            den = self.p_single[:, k].sum()
            for t in range(T):
                num += self.p_single[t,k]*(U[t] - self.mu[k]).reshape(-1, 1).dot((U[t] - self.mu[k]).reshape(1, -1))           
            self.Sigma[k] = num/den                                                                                             

    def _update_mu(self, U):
        for k in range(self.K):
            num = np.multiply(self.p_single[:,k].reshape(-1,1), U).sum(axis=0)
            den = self.p_single[:, k].sum()
            self.mu[k] = num/den

    def _update_A(self, U):
        for k in range(self.K):
            for l in range(self.K):
                self.A[k,l] = self.p_double[1:, k, l].sum()/self.p_single[1:, k].sum()

    def _update_pi(self, U):
        for k in range(self.K):
            self.pi[k] = self.p_single[0, k]/self.p_single[0, :].sum()

    def _update_p_single(self, U):
        T, d = U.shape
        for t in range(T):
            for k in range(self.K):
                self.p_single[t, k] = self._cond_distribution_single(t, k, U)

    def _update_p_double(self, U):
        T, d = U.shape
        for t in range(1, T):
            for k in range(self.K):
                for l in range(self.K):
                    self.p_double[t-1, k, l] = self._cond_distribution_double([t-1,t], [k,l], U)

    def _cond_distribution_single(self, t, q, U):
        return np.exp(self.log_alpha[t,q] + self.log_beta[t, q] - self._log_plus(self.log_alpha[t,:] + self.log_beta[t,:]))

    def _cond_distribution_double(self, t, q, U):
        return np.exp(self.log_alpha[t[0], q[0]] + self.log_beta[t[1], q[1]] + np.log(self.A[q[0], q[1]]) + \
                      self._log_psi_q_u(q[1], U[t[1]]) - self._log_plus(self.log_alpha[t[0], :] + self.log_beta[t[0], :]))

    def _log_psi_q_u(self, q, u):
        d = u.shape[0]
        mu = self.mu[q]
        Sigma = self.Sigma[q]
        c = -d/2.*np.log(2*np.pi) - 1/2.*np.log(np.linalg.det(Sigma))
        return c - 1/2.*((u-mu).reshape(1, -1).dot(np.linalg.inv(Sigma)).dot((u-mu).reshape(-1, 1)))

    def _init_log_alpha(self, U):
        for k in range(self.K):
            self.log_alpha[0, k] = np.log(self.pi[k]) + self._log_psi_q_u(k, U[0])

    def _init_log_beta(self, U):
        for q in range(self.K):
            self.log_beta[-1, q] = np.log(1)

    def _forward_log_alpha(self, U, t):
        for q in range(self.K):
            self.log_alpha[t, q] = self._log_psi_q_u(q, U[t]) + self._log_plus(np.log(self.A[:, q]) + self.log_alpha[t-1,:])

    def _backward_log_beta(self, U, t):
        for q in range(self.K):
            self.log_beta[t, q] = self._log_plus([self._log_psi_q_u(r, U[t+1]) + np.log(self.A[q, r]) + \
                                                 self.log_beta[t+1,r] for r in range(self.K)])

    def _init_parameters(self, U):
        T, d = U.shape
        # Gaussian Mixture Model with full covariance
        clf = FullGMM(k=self.K, max_iter=self.max_iter) 
        clf.fit(U)
        # Initialized covariances, means and labels with output of GMM
        self.mu = clf.mu
        self.Sigma = clf.Sigma
        self.labels = clf.labels
       
        self.pi = np.full(self.K, 1./(self.K)) # Uniform distribution
        self.A = np.full((self.K, self.K), 1./(self.K)) # Uniform transitions

        if self.verbose > 0:
            print("Initialization of parameters with mixture of Gaussian models complete...")

    def _viberti(self, U):
        T, d = U.shape
        # p[k,t] stores probability of most likely state (q_1, ..., q_t-1, q_t=k) for obs u_1, ..., u_t
        # arg[k,t] stores the argmax over k of that probability expressed recursively with p[q, t-1]
        log_p = np.zeros((self.K, T))
        arg = np.zeros((self.K, T)).astype(int)

        # Initialize
        for k in range(self.K):
            log_p[k, 0] = np.log(self.pi[k]) + self._log_psi_q_u(k, U[0])
            arg[k, 0] = 0

        # Use recursive formula
        # p[k,t] = max_x psi_q_u(k,U[t]) A[x, k] p[x, t-1]
        for t in range(1, T):
            for k in range(self.K):
                tmp = [log_p[x, t-1] + np.log(self.A[x, k]) + self._log_psi_q_u(k, U[t]) for x in range(self.K)]
                log_p[k, t] = max(tmp)
                arg[k, t] = np.argmax(tmp)

        labels = np.zeros(T).astype(int)
        labels[T-1] = np.argmax(log_p[:, T-1])
        for t in reversed(range(T-1)):
            labels[t] = arg[labels[t+1], t+1]

        return labels

    def _get_lc(self, U, labels):
        T, d = U.shape
        lc = np.log(self.pi[labels[0]])
        for t in range(1, T):
            lc += np.log(self.A[labels[t-1], labels[t]])
        for t in range(T):
            mu = self.mu[labels[t]]
            Sigma = self.Sigma[labels[t]]
            lc += -d/2.*np.log(np.pi) - 1/2.*np.log(np.linalg.det(Sigma)) - \
                   1/2.*(U[t]-mu).dot(np.linalg.inv(Sigma)).dot(U[t]-mu)
        return lc

    def predict(self, U):
        return self._viberti(U)
