"""Minorization-maximization inference algorithms."""

import itertools
import numpy as np

from .convergence import NormOfDifferenceTest
from .utils import log_transform, exp_transform


def _mm(n_items, data, initial_params, alpha, max_iter, tol, mm_fun):
    """
    Iteratively refine MM estimates until convergence.

    Raises
    ------
    RuntimeError
        If the algorithm does not converge after `max_iter` iterations.
    """
    if initial_params is None:
        params = np.zeros(n_items)
    else:
        params = initial_params
    converged = NormOfDifferenceTest(tol=tol, order=1)
    for _ in range(max_iter):
        nums, denoms = mm_fun(n_items, data, params)
        params = log_transform((nums + alpha) / (denoms + alpha))
        if converged(params):
            return params
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))


def _mm_pairwise(n_items, data, params):
    """Inner loop of MM algorithm for pairwise data."""
    weights = exp_transform(params)
    wins = np.zeros(n_items, dtype=float)
    denoms = np.zeros(n_items, dtype=float)
    for winner, loser in data:
        wins[winner] += 1.0
        val = 1.0 / (weights[winner] + weights[loser])
        denoms[winner] += val
        denoms[loser] += val
    return wins, denoms


def mm_pairwise(
        n_items, data, initial_params=None, alpha=0.0,
        max_iter=10000, tol=1e-8):
    """Compute the ML estimate of model parameters using the MM algorithm.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given pairwise-comparison data (see :ref:`data-pairwise`), using
    the minorization-maximization (MM) algorithm [Hun04]_, [CD12]_.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise-comparison data.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Maximum L1-norm of the difference between successive iterates to
        declare convergence.

    Returns
    -------
    params : numpy.ndarray
        The ML estimate of model parameters.
    """
    return _mm(
            n_items, data, initial_params, alpha, max_iter, tol, _mm_pairwise)


def _mm_rankings(n_items, data, params):
    """Inner loop of MM algorithm for ranking data."""
    weights = exp_transform(params)
    wins = np.zeros(n_items, dtype=float)
    denoms = np.zeros(n_items, dtype=float)
    for ranking in data:
        sum_ = weights.take(ranking).sum()
        for i, winner in enumerate(ranking[:-1]):
            wins[winner] += 1
            val = 1.0 / sum_
            for item in ranking[i:]:
                denoms[item] += val
            sum_ -= weights[winner]
    return wins, denoms


def mm_rankings(n_items, data, initial_params=None, alpha=0.0,
        max_iter=10000, tol=1e-8):
    """Compute the ML estimate of model parameters using the MM algorithm.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given ranking data (see :ref:`data-rankings`), using the
    minorization-maximization (MM) algorithm [Hun04]_, [CD12]_.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Ranking data.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Maximum L1-norm of the difference between successive iterates to
        declare convergence.

    Returns
    -------
    params : numpy.ndarray
        The ML estimate of model parameters.
    """
    return _mm(n_items, data, initial_params, alpha, max_iter, tol,
            _mm_rankings)


def _mm_top1(n_items, data, params):
    """Inner loop of MM algorithm for top1 data."""
    weights = exp_transform(params)
    wins = np.zeros(n_items, dtype=float)
    denoms = np.zeros(n_items, dtype=float)
    for winner, losers in data:
        wins[winner] += 1
        val = 1 / (weights.take(losers).sum() + weights[winner])
        for item in itertools.chain([winner], losers):
            denoms[item] += val
    return wins, denoms


def mm_top1(
        n_items, data, initial_params=None, alpha=0.0,
        max_iter=10000, tol=1e-8):
    """Compute the ML estimate of model parameters using the MM algorithm.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given top-1 data (see :ref:`data-top1`), using the
    minorization-maximization (MM) algorithm [Hun04]_, [CD12]_.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Top-1 data.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Maximum L1-norm of the difference between successive iterates to
        declare convergence.

    Returns
    -------
    params : numpy.ndarray
        The ML estimate of model parameters.
    """
    return _mm(n_items, data, initial_params, alpha, max_iter, tol, _mm_top1)


def _choicerank(n_items, data, params):
    """Inner loop of ChoiceRank algorithm."""
    weights = exp_transform(params)
    adj, adj_t, traffic_in, traffic_out = data
    # First phase of message passing.
    zs = adj.dot(weights)
    # Second phase of message passing.
    with np.errstate(invalid="ignore"):
        denoms = adj_t.dot(traffic_out / zs)
    return traffic_in, denoms


def choicerank(
        digraph, traffic_in, traffic_out, weight=None,
        initial_params=None, alpha=1.0, max_iter=10000, tol=1e-8):
    """Compute the MAP estimate of a network choice model's parameters.

    This function computes the maximum-a-posteriori (MAP) estimate of model
    parameters given a network structure and node-level traffic data (see
    :ref:`data-network`), using the ChoiceRank algorithm [MG17]_, [KTVV15]_.

    The nodes are assumed to be labeled using consecutive integers starting
    from 0.

    Parameters
    ----------
    digraph : networkx.DiGraph
        Directed graph representing the network.
    traffic_in : array_like
        Number of arrivals at each node.
    traffic_out : array_like
        Number of departures at each node.
    weight : str, optional
        The edge attribute that holds the numerical value used for the edge
        weight. If None (default) then all edge weights are 1.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Maximum L1-norm of the difference between successive iterates to
        declare convergence.

    Returns
    -------
    params : numpy.ndarray
        The MAP estimate of model parameters.

    Raises
    ------
    ImportError
        If the NetworkX library cannot be imported.
    """
    import networkx as nx
    # Compute the (sparse) adjacency matrix.
    n_items = len(digraph)
    nodes = np.arange(n_items)
    adj = nx.to_scipy_sparse_matrix(digraph, nodelist=nodes, weight=weight)
    adj_t = adj.T.tocsr()
    # Process the data into a standard form.
    traffic_in = np.asarray(traffic_in)
    traffic_out = np.asarray(traffic_out)
    data = (adj, adj_t, traffic_in, traffic_out)
    return _mm(
            n_items, data, initial_params, alpha, max_iter, tol, _choicerank)
