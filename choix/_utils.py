import math
import numpy as np
import random
import scipy.linalg as spl
import warnings

from scipy.linalg import solve_triangular


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def normcdf(x):
    """Normal cumulative density function."""
    # If X ~ N(0,1), returns P(X < x).
    return math.erfc(-x / SQRT2) / 2.0


def normpdf(x):
    """Normal probability density function."""
    return math.exp(-x*x / 2.0) / SQRT2PI


def inv_pd(mat):
    """Stable inverse of a positive definite matrix."""
    # See:
    # - http://www.seas.ucla.edu/~vandenbe/103/lectures/chol.pdf
    # - http://scicomp.stackexchange.com/questions/3188
    chol = np.linalg.cholesky(mat)
    ident = np.eye(mat.shape[0])
    res = solve_triangular(chol, ident, lower=True, overwrite_b=True)
    return np.transpose(res).dot(res)


def log_likelihood_rankings(rankings, strengths):
    """Compute the log-likelihood of Plackett--Luce model parameters.
    Args:
        rankings (List[tuple]): The data (partial rankings.)
        strengths (List[float]): The model parameters.
    Returns:
        loglik (float): the log-likelihood of the parameters given the data.
    """
    loglik = 0
    for ranking in rankings:
        sum_ = sum(strengths[x] for x in ranking)
        for i, winner in enumerate(ranking[:-1]):
            loglik += math.log(strengths[winner])
            loglik -= math.log(sum_)
            sum_ -= strengths[winner]
    return loglik


def statdist(generator):
    """Compute the stationary distribution of a Markov chain.
    Args:
        generator (numpy.ndarray): The infinitesimal generator matrix of the
            Markov chain.
    Returns:
        dist (List[float]): The unnormalized stationary distribution of the
            Markov chain.
    """
    n = generator.shape[0]
    with warnings.catch_warnings():
        # The LU decomposition raises a warning when the generator matrix is
        # singular (which it, by construction, is!).
        warnings.filterwarnings('ignore')
        lu, piv = spl.lu_factor(generator.T, check_finite=False)
    # The last row contains 0's only.
    left = lu[:-1,:-1]
    right = -lu[:-1,-1]
    # Solves system `left * x = right`. Assumes that `left` is
    # upper-triangular (ignores lower triangle).
    res = spl.solve_triangular(left, right, check_finite=False)
    res = np.append(res, 1.0)
    return (n / res.sum()) * res


def generate_rankings(strengths, nb_rankings, size_of_ranking=3):
    """Generate random rankings according to a Plackett--Luce model.
    Args:
        strengths (List[float]): The model parameters.
        nb_rankings (int): The number of rankings to generate.
        size_of_ranking (Optional[int]): The number of items to include in each
            ranking. Default value: 3.
    Returns:
        data (List[tuple]): A list of (partial) rankings generated according to
            a Plackett--Luce model with the specified model parameters.
    """
    n = len(strengths)
    items = range(n)
    data = list()
    for _ in range(nb_rankings):
        alts = random.sample(items, size_of_ranking)
        probs = np.array([strengths[x] for x in alts])
        datum = list()
        for _ in range(size_of_ranking):
            probs /= np.sum(probs)
            idx = np.random.choice(size_of_ranking, p=probs)
            datum.append(alts[idx])
            probs[idx] = 0.0
        data.append(tuple(datum))
    return tuple(data)
