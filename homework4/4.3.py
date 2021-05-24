# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 21:55:21 2021

@author: rahul
"""
import numpy as np
pi = np.pi

X = [-1, 0, 4, 5, 6]
pjs = np.array([.5, .5], dtype=np.float64)
mujs = np.array([6, 7], dtype=np.float64)
varjs = np.array([1, 4], dtype=np.float64)

theta = [pjs, mujs, varjs]
P_ji = {}
P_xi = {}


def norm(mu, var, x):
    return (1/(2*pi*var))*np.exp(-((x-mu)**2)/(2*var))


for i in range(5):
    r = 0
    for j in range(2):
        r += pjs[j]*norm(mujs[j], varjs[j], X[i])
    P_xi[i] = r


def prob(j, i):
    return (pjs[j]*norm(mujs[j], varjs[j], X[i]))/P_xi[i]


def likelihood(X, theta):
    log_likelihood = 0
    for i in range(5):
        t = 0
        for j in range(2):
            mu = mujs[j]
            var = varjs[j]
            p = pjs[j]
            t += p*norm(mu, var, X[i])
        log_likelihood += np.log(t)
        return log_likelihood
print(likelihood(X, theta))


def estep():
    for j in range(2):
        for i in range(5):
            P_ji[j,i] = prob(j,i)


def mstep():
    for j in range(2):
        s = 0
        d = 0
        for i in range(5):
            s += X[i]*P_ji[j,i]
            d += P_ji[j,i]
        mujs[j] = s/d
        pjs[j] = d/5

        u = 0
        for i in range(5):
            u += P_ji[j,i]*(X[i]-mujs[j])**2
        varjs[j] = u/d


def em():
    for i in range(10):
        estep()
        mstep()
        print(mujs, varjs)

em()
