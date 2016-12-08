import math
import numpy as np
import random
import scipy.linalg as spl
import warnings

from scipy.linalg import solve_triangular
from scipy.stats import rankdata


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


def footrule_dist(params1, params2=None):
    r"""Compute Spearman's footrule distance between two models.

    This function computes Spearman's footrule distance between the rankings
    induced by the two parameter vectors. Let :math:`\sigma_i` be the rank of
    item ``i`` in the first model, and :math:`\tau_i` be its rank in the second
    model. Spearman's footrule distance is defined by

    .. math::

      \sum_{i=1}^N | \sigma_i - \tau_i |

    If the argument ``params2`` is ``None``, the second model is assumed to
    rank the items by their index: item ``0`` has rank 1, item ``1`` has rank
    2, etc.

    Parameters
    ----------
    params1 : array_like
        Parameters of the first model.
    params2 : array_like, optional
        Parameters of the second model.

    Returns
    -------
    dist : float
        Spearman's footrule distance.
    """
    assert params2 is None or len(params1) == len(params2)
    # We use `-params` because the highest values should be ranked first.
    ranks1 = rankdata(-params1, method="average")
    if params2 is None:
        ranks2 = np.arange(1, len(params1) + 1, dtype=float)
    else:
        ranks2 = rankdata(-params2, method="average")
    return np.sum(np.abs(ranks1 - ranks2))


def log_likelihood_pairwise(data, params):
    """Compute the log-likelihood of model parameters."""
    loglik = 0
    for winner, loser in data:
        loglik += math.log(params[winner])
        loglik -= math.log(params[winner] + params[loser])
    return loglik


def log_likelihood_rankings(data, params):
    """Compute the log-likelihood of model parameters."""
    loglik = 0
    params = np.asarray(params)
    for ranking in data:
        sum_ = params.take(ranking).sum()
        for i, winner in enumerate(ranking[:-1]):
            loglik += math.log(params[winner])
            loglik -= math.log(sum_)
            sum_ -= params[winner]
    return loglik


def log_likelihood_top1(data, params):
    """Compute the log-likelihood of model parameters."""
    loglik = 0
    for winner, losers in data:
        loglik += math.log(params[winner])
        loglik -= math.log(params[winner] + params.take(losers).sum())
    return loglik


def statdist(generator):
    """Compute the stationary distribution of a Markov chain.

    Parameters
    ----------
    generator : array_like
        Infinitesimal generator matrix of the Markov chain.

    Returns
    -------
    dist : np.array
        The unnormalized stationary distribution of the Markov chain.

    Raises
    ------
    ValueError
        If the Markov chain does not have a unique stationary distribution.
    """
    generator = np.asarray(generator)
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
    try:
        res = spl.solve_triangular(left, right, check_finite=False)
    except:
        # Ideally we would like to catch `spl.LinAlgError` only, but there seems
        # to be a bug in scipy, in the code that raises the LinAlgError (!!).
        raise ValueError("stationary distribution could not be computed."
                "Perhaps the Markov chain has more than one absorbing class?")
    res = np.append(res, 1.0)
    return (n / res.sum()) * res


def generate_pairwise(params, num_comparisons=10):
    """Generate pairwise comparisons from a Bradley--Terry model.

    This function samples comparisons pairs independently and uniformly at
    random over the ``len(params)`` choose 2 possibilities, and samples the
    corresponding comparison outcomes from a Bradley--Terry model parametrized
    by ``params``.

    Parameters
    ----------
    params : array_like
        The parameters of the Bradley--Terry model.
    num_comparisons : int
        The number of comparisons to be returned.

    Returns
    -------
    data : list of (int, int)
       The samples (see :ref:`data-pairwise`).
    """
    n = len(params)
    items = tuple(range(n))
    params = np.asarray(params)
    data = list()
    for _ in range(num_comparisons):
        # Pick the pair uniformly at random.
        a, b = random.sample(items, 2)
        if compare((a, b), params) == a:
            data.append((a, b))
        else:
            data.append((b, a))
    return tuple(data)


def generate_rankings(params, num_rankings, size=3):
    """Generate rankings according to a Plackett--Luce model.

    This function samples subsets of items (of size ``size``) independently and
    uniformly at random, and samples the correspoding partial ranking from a
    Plackett--Luce model parametrized by ``params``.

    Parameters
    ----------
    params : array_like
        Model parameters.
    num_rankings : int
        Number of rankings to generate.
    size : int, optional
        Number of items to include in each ranking.

    Returns
    -------
    data : list of np.array
        A list of (partial) rankings generated according to a Plackett--Luce
        model with the specified model parameters.
    """
    n = len(params)
    items = tuple(range(n))
    params = np.asarray(params)
    data = list()
    for _ in range(num_rankings):
        # Pick the alternatives uniformly at random.
        alts = random.sample(items, size)
        ranking = compare(alts, params, rank=True)
        data.append(ranking)
    return tuple(data)


def compare(items, params, rank=False):
    """Generate a comparison outcome that follows Luce's axiom.

    This function samples an outcome for the comparison of a subset of items,
    from a model parametrized by ``params``. If ``rank`` is True, it returns a
    ranking over the items, otherwise it returns a single item.

    Parameters
    ----------
    items : list
        The subset of items to compare.
    params : array_like
        The ``N`` parameters of the model.
    rank : bool, optional
        If true, returns a ranking over the items instead of a single item.

    Returns
    -------
    outcome : int or list of int
        The chosen item, or a ranking over ``items``.
    """
    params = np.asarray(params)
    probs = params.take(items)
    probs /= probs.sum()
    if rank:
        return np.random.choice(items, size=len(items), replace=False, p=probs)
    else:
        return np.random.choice(items, p=probs)
