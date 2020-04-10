
import numpy as np
import scipy as sc
from scipy.stats import gamma, beta, norm, uniform, expon
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, Rbf
from scipy.optimize import minimize, newton

from epidemiccore_w import *

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

def draw_from_prior(p):
    return [p[0].rvs(), p[1].rvs(), p[2].rvs()]

def prior(theta,p):
    return p[0].pdf(theta[0]) * p[1].pdf(theta[2]) * p[2].pdf(theta[2])

def log_prior(theta,p):
    if prior(theta,p) > 0:
        return np.log(prior(theta,p))
    else:
        return -np.inf


def log_data_likelihood(epi, theta):
    a,b,c=theta
    S0 = [1.0]
    #t = np.linspace(0,self.data.max(),1000)
    sol = odeint(Epidemic.Deriv_S, S0, epi.t, args=(a, b, c))
    S = interp1d(epi.t,sol[:,0])
    smax = S(epi.T)
    factor = 1 - smax
    Z = 0
    j = 0
    for x in epi.data.values:
        s = S(x)
        if s > 0:
            z = (a*s*np.log(s)+b*(s-s**2)+c*s)/factor
            if z > 0:
                Z += np.log(z)
                j += 1
    return Z

def log_posterior(epi,theta,p):
    data_likelihood = log_data_likelihood(epi,theta)
    lp = log_prior(theta, p)
    return data_likelihood + lp

def proposal_loglikelihood(theta1,theta2,q):
    return np.sum(np.log(q.pdf(theta2)))

def draw_proposal(theta, proposal):
    ss = proposal.rvs(size=len(theta)) + theta
    return ss

def accept_prob(epi, p, theta1, theta2):
    l1 = log_posterior(epi,theta1,p)
    l2 = log_posterior(epi,theta2,p)
    a = min(1, np.exp(l2 -l1))
    return a

def mh(epi, p, proposal, burnin=10**3, chain_length=10**4):
    chain = np.zeros((chain_length,3), dtype=np.float)
    chain[0] = draw_from_prior(p)
    for i in np.arange(1,chain_length):
        new = draw_proposal(chain[i-1], proposal)
        a = accept_prob(epi, p, chain[i-1], new)
        if uniform.rvs() < a:
            chain[i] = new
        else:
            chain[i] = chain[i-1]
    after_burnin = chain[burnin:chain_length-1]
    idx = np.arange(0,chain_length-burnin,100)
    return after_burnin[idx]



