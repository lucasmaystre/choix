import math
import numpy as np
import random
import scipy.linalg as spl
import warnings

from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.stats import rankdata, kendalltau


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def log_transform(weights):
    """Transform weights into centered log-scale parameters."""
    params = np.log(weights)
    return params - params.mean()


def exp_transform(params):
    """Transform parameters into exp-scale weights."""
    weights = np.exp(np.asarray(params) - np.mean(params))
    return (len(weights) / weights.sum()) * weights


def softmax(xs):
    """Stable implementation of the softmax function."""
    ys = xs - np.max(xs)
    exps = np.exp(ys)
    return exps / exps.sum(axis=0)


def normal_cdf(x):
    """Normal cumulative density function."""
    # If X ~ N(0,1), returns P(X < x).
    return math.erfc(-x / SQRT2) / 2.0


def normal_pdf(x):
    """Normal probability density function."""
    return math.exp(-x*x / 2.0) / SQRT2PI


def inv_posdef(mat):
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
    induced by two parameter vectors. Let :math:`\sigma_i` be the rank of item
    ``i`` in the model described by ``params1``, and :math:`\tau_i` be its rank
    in the model described by ``params2``. Spearman's footrule distance is
    defined by

    .. math::

      \sum_{i=1}^N | \sigma_i - \tau_i |

    By convention, items with the lowest parameters are ranked first (i.e.,
    sorted using the natural order).

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
    ranks1 = rankdata(params1, method="average")
    if params2 is None:
        ranks2 = np.arange(1, len(params1) + 1, dtype=float)
    else:
        ranks2 = rankdata(params2, method="average")
    return np.sum(np.abs(ranks1 - ranks2))


def kendalltau_dist(params1, params2=None):
    r"""Compute the Kendall tau distance between two models.

    This function computes the Kendall tau distance between the rankings
    induced by two parameter vectors. Let :math:`\sigma_i` be the rank of item
    ``i`` in the model described by ``params1``, and :math:`\tau_i` be its rank
    in the model described by ``params2``. The Kendall tau distance is defined
    as the number of pairwise disagreements between the two rankings, i.e.,

    .. math::

      \sum_{i=1}^N \sum_{j=1}^N
        \mathbf{1} \{ \sigma_i > \sigma_j \wedge \tau_i < \tau_j \}

    By convention, items with the lowest parameters are ranked first (i.e.,
    sorted using the natural order).

    If the argument ``params2`` is ``None``, the second model is assumed to
    rank the items by their index: item ``0`` has rank 1, item ``1`` has rank
    2, etc.

    If some values are equal within a parameter vector, all items are given a
    distinct rank, corresponding to the order in which the values occur.

    Parameters
    ----------
    params1 : array_like
        Parameters of the first model.
    params2 : array_like, optional
        Parameters of the second model.

    Returns
    -------
    dist : float
        Kendall tau distance.
    """
    assert params2 is None or len(params1) == len(params2)
    ranks1 = rankdata(params1, method="ordinal")
    if params2 is None:
        ranks2 = np.arange(1, len(params1) + 1, dtype=float)
    else:
        ranks2 = rankdata(params2, method="ordinal")
    tau, _ = kendalltau(ranks1, ranks2)
    n_items = len(params1)
    n_pairs = n_items * (n_items - 1) / 2
    return round((n_pairs - n_pairs * tau) / 2)


def rmse(params1, params2):
    r"""Compute the root-mean-squared error between two models.

    Parameters
    ----------
    params1 : array_like
        Parameters of the first model.
    params2 : array_like
        Parameters of the second model.

    Returns
    -------
    error : float
        Root-mean-squared error.
    """
    assert len(params1) == len(params2)
    params1 = np.asarray(params1) - np.mean(params1)
    params2 = np.asarray(params2) - np.mean(params2)
    sqrt_n = math.sqrt(len(params1))
    return np.linalg.norm(params1 - params2, ord=2) / sqrt_n


def log_likelihood_pairwise(data, params):
    """Compute the log-likelihood of model parameters."""
    loglik = 0
    for winner, loser in data:
        loglik -= np.logaddexp(0, -(params[winner] - params[loser]))
    return loglik


def log_likelihood_rankings(data, params):
    """Compute the log-likelihood of model parameters."""
    loglik = 0
    params = np.asarray(params)
    for ranking in data:
        for i, winner in enumerate(ranking[:-1]):
            loglik -= logsumexp(params.take(ranking[i:]) - params[winner])
    return loglik


def log_likelihood_top1(data, params):
    """Compute the log-likelihood of model parameters."""
    loglik = 0
    params = np.asarray(params)
    for winner, losers in data:
        idx = np.append(winner, losers)
        loglik -= logsumexp(params.take(idx) - params[winner])
    return loglik


def log_likelihood_network(
        digraph, traffic_in, traffic_out, params, weight=None):
    """
    Compute the log-likelihood of model parameters.

    If ``weight`` is not ``None``, the log-likelihood is correct only up to a
    constant (independent of the parameters).
    """
    loglik = 0
    for i in range(len(traffic_in)):
        loglik += traffic_in[i] * params[i]
        if digraph.out_degree(i) > 0:
            neighbors = list(digraph.successors(i))
            if weight is None:
                loglik -= traffic_out[i] * logsumexp(params.take(neighbors))
            else:
                weights = [digraph[i][j][weight] for j in neighbors]
                loglik -= traffic_out[i] * logsumexp(
                        params.take(neighbors), b=weights)
    return loglik


def statdist(generator):
    """Compute the stationary distribution of a Markov chain.

    Parameters
    ----------
    generator : array_like
        Infinitesimal generator matrix of the Markov chain.

    Returns
    -------
    dist : numpy.ndarray
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
        warnings.filterwarnings("ignore")
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
        raise ValueError(
                "stationary distribution could not be computed. "
                "Perhaps the Markov chain has more than one absorbing class?")
    res = np.append(res, 1.0)
    return (n / res.sum()) * res


def generate_params(n_items, interval=5.0, ordered=False):
    r"""Generate random model parameters.

    This function samples a parameter independently and uniformly for each
    item. ``interval`` defines the width of the uniform distribution.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    interval : float
        Sampling interval.
    ordered : bool, optional
        If true, the parameters are ordered from lowest to highest.

    Returns
    -------
    params : numpy.ndarray
       Model parameters.
    """
    params = np.random.uniform(low=0, high=interval, size=n_items)
    if ordered:
        params.sort()
    return params - params.mean() 


def generate_pairwise(params, n_comparisons=10):
    """Generate pairwise comparisons from a Bradley--Terry model.

    This function samples comparisons pairs independently and uniformly at
    random over the ``len(params)`` choose 2 possibilities, and samples the
    corresponding comparison outcomes from a Bradley--Terry model parametrized
    by ``params``.

    Parameters
    ----------
    params : array_like
        Model parameters.
    n_comparisons : int
        Number of comparisons to be returned.

    Returns
    -------
    data : list of (int, int)
       Pairwise-comparison samples (see :ref:`data-pairwise`).
    """
    n = len(params)
    items = tuple(range(n))
    params = np.asarray(params)
    data = list()
    for _ in range(n_comparisons):
        # Pick the pair uniformly at random.
        a, b = random.sample(items, 2)
        if compare((a, b), params) == a:
            data.append((a, b))
        else:
            data.append((b, a))
    return tuple(data)


def generate_rankings(params, n_rankings, size=3):
    """Generate rankings according to a Plackett--Luce model.

    This function samples subsets of items (of size ``size``) independently and
    uniformly at random, and samples the correspoding partial ranking from a
    Plackett--Luce model parametrized by ``params``.

    Parameters
    ----------
    params : array_like
        Model parameters.
    n_rankings : int
        Number of rankings to generate.
    size : int, optional
        Number of items to include in each ranking.

    Returns
    -------
    data : list of numpy.ndarray
        A list of (partial) rankings generated according to a Plackett--Luce
        model with the specified model parameters.
    """
    n = len(params)
    items = tuple(range(n))
    params = np.asarray(params)
    data = list()
    for _ in range(n_rankings):
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
        Subset of items to compare.
    params : array_like
        Model parameters.
    rank : bool, optional
        If true, returns a ranking over the items instead of a single item.

    Returns
    -------
    outcome : int or list of int
        The chosen item, or a ranking over ``items``.
    """
    probs = probabilities(items, params)
    if rank:
        return np.random.choice(items, size=len(items), replace=False, p=probs)
    else:
        return np.random.choice(items, p=probs)


def probabilities(items, params):
    """Compute the comparison outcome probabilities given a subset of items.

    This function computes, for each item in ``items``, the probability that it
    would win (i.e., be chosen) in a comparison involving the items, given
    model parameters.

    Parameters
    ----------
    items : list
        Subset of items to compare.
    params : array_like
        Model parameters.

    Returns
    -------
    probs : numpy.ndarray
        A probability distribution over ``items``.
    """
    params = np.asarray(params)
    return softmax(params.take(items))
