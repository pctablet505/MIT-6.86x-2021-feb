"""Mixture model for matrix completion"""



from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    eps = 1e-16
    pi = np.pi
    n, d = X.shape
    Mu, Var, Pi = mixture  # k*d, k*1, k*1

    K = len(Var)
    delta = np.where(X != 0, 1, 0)  # n*d
    F = np.zeros((n, K))  # n*k

    F += np.log(Pi+eps)
    F -= (delta.sum(axis=1, keepdims=True) @
          np.log(2*pi*Var+eps).reshape((1, K))/2)
    F -= ((X**2).sum(axis=1)[:, None]+(delta@Mu.T**2)-2*X@Mu.T)/(2*Var+eps)

    max_f = F.max(axis=1, keepdims=True)

    log_likelihood = (max_f+logsumexp(F-max_f, axis=1, keepdims=True)).sum()

    log_post = F-(max_f+logsumexp(F-max_f, axis=1, keepdims=True))
    post = np.exp(log_post)
    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    mu, var, p = mixture
    K = len(p)
    n, d = X.shape
    eps = 1e-16

    # n,d
    delta = np.where(X != 0, 1, 0)
    p_hat = np.sum(post, axis=0)/n

    '''
    #Loop version
    munr = np.zeros((K, d))
    mudr = np.zeros((K, d))
    for u in range(n):
        munr += post[u].reshape(K, 1)@((delta[u]*X[u]).reshape(1, d))
        mudr += (post[u].reshape(K, 1)@(delta[u].reshape(1, d)))
    '''
    ############## Vectorized version ###############

    logpost = np.log(post+eps)
    logmunr = logsumexp((logpost[:, None].transpose(2, 0, 1) +
                         np.log(X+eps)+np.log(delta+eps)), axis=1)
    logmudr = logsumexp((logpost[:, None].transpose(
        2, 0, 1)+np.log(delta+eps)), axis=1)
    logmu = logmunr-logmudr
    mu_new = np.exp(logmu)

    ################################################

    # only to update mu where mudr>=1
    mu_hat = np.where(logmudr > 0, mu_new, mu)

    var_hat = (post*(np.sum(X**2, axis=1)[:, None] +
                      (delta@mu_hat.T**2)-2*X@mu_hat.T)).sum(axis=0)

    var_hat = var_hat / np.exp(logmudr).sum(axis=1)
    

    var_hat = np.maximum(var_hat, min_variance)

    return GaussianMixture(mu_hat, var_hat, p_hat)


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

    while prev_cost is None or cost-prev_cost > (1e-6*abs(cost)):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post, mixture, min_variance=.25)

    return mixture, post, cost


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """

    post, _ = estep(X, mixture)
    X_hat = (post@mixture.mu)
    return np.where(X != 0, X, X_hat)
