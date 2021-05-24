"""Mixture model using EM"""
import numpy as np
from typing import Tuple

from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = len(mixture.var)
    post = np.zeros((n, K))
    pi = np.pi
    norm = np.linalg.norm
    log_likelihood = 0

    def normal(mu, var, x):
        return 1 / ((2 * pi * var) ** (d / 2)) * np.exp(-(norm(x - mu) ** 2) / (2 * var))

    for i in range(n):
        t = 0
        for j in range(K):
            mu = mixture.mu[j]
            var = mixture.var[j]
            p = mixture.p[j]
            post[i][j] = p * normal(mu, var, X[i])
            t += p * normal(mu, var, X[i])
        log_likelihood += np.log(t)
    # normalizing the probablities
    for i in range(n):
        post[i, :] /= post[i].sum()
    
    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    norm = np.linalg.norm
    n, d = X.shape
    n, K = post.shape
    mujs = []
    pjs = []
    varjs = []
    for j in range(K):
        mujs.append(np.dot(post[:, j], X)/post[:, j].sum())
        pjs.append(post[:, j].sum()/n)
        mu = mujs[-1]
        varjs.append(np.dot(post[:, j], norm(
            X-mu, axis=1)**2)/(d*post[:, j].sum()))
    mujs = np.array(mujs)
    pjs = np.array(pjs)
    varjs = np.array(varjs)
    return GaussianMixture(mujs, varjs, pjs)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    n, K = post.shape

    prev_cost = None
    cost = None
    while prev_cost is None or abs(cost-prev_cost) > (10**(-6)*abs(cost)):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, cost
