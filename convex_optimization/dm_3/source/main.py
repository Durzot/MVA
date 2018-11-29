import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Oracle(object):
    """
    Object that takes as input the parameters below that define the f function.
    The object implements methods to evalutate the function, its gradient and its hessian 
    at any admissible point v

    Q: 2D np.array (n, n)
    p: 1D np.array (n,)
    A: 2D np.array (d, n)
    b: 1D np.array (d,)
    t: float
    v: 1D np.array (n,)
    """
    def __init__(self, Q, p, A, b, t):
        self.Q = Q
        self.p = p
        self.A = A
        self.b = b
        self.t = t

    def eval(self, v):
        d, n = self.A.shape
        f1 = self.t*(v.dot(self.Q).dot(v) + v.dot(self.p))
        f2 = 0
        for i in range(d):
            f2 += - np.log(self.b[i]-self.A[i,:].dot(v))
        return f1+f2

    def eval_f0(self, v):
        return v.dot(self.Q).dot(v) + v.dot(self.p)

    def grad(self, v):
        d, n = self.A.shape
        g1 = self.t*(2*self.Q.dot(v) + self.p)
        g2 = np.zeros(n)
        for i in range(d):
            g2 += self.A[i,:]/(self.b[i]-self.A[i,:].dot(v))
        return g1+g2
    
    def hess(self, v):
        d, n = self.A.shape
        h1 = 2*self.t*self.Q
        h2 = (self.A.T/(self.b-self.A.dot(v))**2).dot(self.A)
        return h1+h2


def backtrack_search(oracle, v, delta_v, s0, alpha, beta, max_iter=50000):
    s = s0
    f_v = oracle.eval(v)
    grad_v = oracle.grad(v)
    f_v_delta_v = oracle.eval(v+s*delta_v)

    compt = 1
    while f_v_delta_v >= f_v + alpha*s*grad_v.dot(delta_v) or np.isnan(f_v_delta_v):
        compt += 1
        s = beta*s
        f_v_delta_v = oracle.eval(v+s*delta_v)
        if compt == max_iter:
            print("max_iter = %d reached in backtrack searck algorithm" % max_iter)
            break
    return (s, compt)

def centering_step(Q, p, A, b, t, v0, eps, alpha=0.25, beta=0.75):
    oracle = Oracle(Q, p, A, b, t)
    v = v0
    lbda_2 = 2*eps
    compt_newton = 0
    compt = 0

    while lbda_2 >= 2*eps:
        delta_v = - np.linalg.inv(oracle.hess(v)).dot(oracle.grad(v))
        lbda_2 = -oracle.grad(v).dot(delta_v)
        s, c = backtrack_search(oracle, v, delta_v, 1, alpha, beta)
        v += s*delta_v
        compt += 1
        compt_newton += c

    return (oracle.eval_f0(v), v, compt_newton/compt)

def barr_method(Q, p, A, b, v0, t0, eps, mu):
    d, n = A.shape
    v, t  = v0, t0
    seq_v, seq_f0_v, seq_cmoy_newton = [], [], []
    
    while d/t >= eps:
        # We use the same epsilon for the centering step as for the barrier method
        # We update v by the last value returned by the centering_step function
        f0_v, v, c = centering_step(Q, p, A, b, t, v, eps)
        seq_v.append(v)
        seq_f0_v.append(f0_v)
        seq_cmoy_newton.append(c)
        t = mu*t

    return (np.array(seq_v), np.array(seq_f0_v), np.array(seq_cmoy_newton))


def plot_barr_method(seq_f0_v, seq_cmoy_newton, mu, ax):
    n_eps = seq_f0_v.shape[0]
    ax[0].plot(np.arange(1, n_eps), seq_f0_v[:-1]-seq_f0_v[-1], label="$\mu$ = %d ($f_0^* = %.5g$)" % (mu, seq_f0_v[-1]))
    ax[0].scatter(np.arange(1, n_eps), seq_f0_v[:-1]-seq_f0_v[-1], marker="+")

    ax[1].plot(np.arange(1, n_eps+1), seq_cmoy_newton, label="$\mu$ = %d" % mu)
    ax[1].scatter(np.arange(1, n_eps+1),  seq_cmoy_newton, marker="+")
    print(min(seq_cmoy_newton))

"""
Main code
"""

seed = 23
np.random.seed(seed)

n, d = 100, 50

# The design matrix and the target vector in the LASSO problem
X = np.random.randn(n, d)
y = np.random.randn(n)
lbda = 10

Q = 0.5*np.eye(n)
p = y
A = X.T
b = np.array([1./lbda]*d)

# Generate an initial feasible point
v0 = np.random.rand(n)*0.001
while any([b[i]-A[i,:].dot(v0) <= 0 for i in range(d)]) : 
    v0 = np.random.rand(n)*0.001

t0 = 1
eps = 1e-6

# Plot results for different values of mu
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,16))
for mu in [2, 10, 15, 50, 100, 250, 500]:
    print("Doing mu = %d ..." % mu)
    seq_v, seq_f0_v, seq_cmoy_newton = barr_method(Q, p, A, b, v0, t0, eps, mu)
    plot_barr_method(seq_f0_v, seq_cmoy_newton, mu, ax)
    print("mu = %d done\n" % mu)

ax[0].set_xlabel("Iterations barrier method", fontsize=20)
ax[0].set_ylabel("$f_0(v_t) - f_0^*$", fontsize=25)
ax[0].set_yscale("log")
ax[0].set_title("Barrier method with different values for parameter $\mu$", fontsize=25, fontweight="bold")

ax[1].set_xlabel("Iterations barrier method", fontsize=20)
ax[1].set_ylabel("Average iterations Newton", fontsize=25)
ax[1].set_yscale("log")
ax[1].set_title("Centering step using Newton algorithm", fontsize=25, fontweight="bold")

ax[0].legend(loc="upper right", fontsize=15)
ax[1].legend(loc="upper right", fontsize=15)
plt.savefig("barr.png")


