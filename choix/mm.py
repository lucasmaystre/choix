"""Minorization-maximization inference algorithms."""

import numpy as np

from .convergence import NormOfDifferenceTest


def _mm(n_items, data, initial_params, alpha, max_iter, tol, mm_fun):
    """Iteratively refine MM estimates until convergence."""
    if initial_params is None:
        params = np.ones(n_items)
    else:
        params = initial_params
    converged = NormOfDifferenceTest(tol=tol, order=1)
    for _ in range(max_iter):
        nums, denoms = mm_fun(n_items, data, params)
        params = (nums + alpha) / (denoms + alpha)
        params = (n_items / params.sum()) * params
        if converged(params):
            return params
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))


def _mm_pairwise(n_items, data, params):
    """Inner loop of MM algorithm for pairwise data."""
    wins = np.zeros(n_items, dtype=float)
    denoms = np.zeros(n_items, dtype=float)
    for winner, loser in data:
        wins[winner] += 1.0
        val = 1.0 / (params[winner] + params[loser])
        denoms[winner] += val
        denoms[loser] += val
    return wins, denoms


def mm_pairwise(n_items, data, initial_params=None, alpha=0.0,
        max_iter=10000, tol=1e-8):
    """Compute the ML estimate of model parameters using the MM algorithm.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given pairwise comparison data (see :ref:`data-pairwise`), using
    the minorization-maximization (MM) algorithm [Hun04]_, [CD12]_.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise comparison data.
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
    params : np.array
        The ML estimate of model parameters.
    """
    return _mm(n_items, data, initial_params, alpha, max_iter, tol,
            _mm_pairwise)
