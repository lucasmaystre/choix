"""(Iterative) Luce Spectral Ranking inference algorithms."""
import numpy as np

from .convergence import NormOfDifferenceTest
from .utils import statdist


def _init_lsr(n_items, alpha, initial_params):
    """Initialize the LSR Markov chain and the weights."""
    if initial_params is None:
        ws = np.ones(n_items)
    else:
        ws = np.asarray(initial_params)
    chain = alpha * np.ones((n_items, n_items), dtype=float)
    return ws, chain


def _ilsr(n_items, data, alpha, params, max_iter, tol, lsr_fun):
    """Iteratively refine LSR estimates until convergence.

    Raises
    ------
    RuntimeError
        If the algorithm does not converge after `max_iter` iterations.
    """
    converged = NormOfDifferenceTest(tol, order=1)
    for _ in range(max_iter):
        params = lsr_fun(n_items, data, alpha=alpha, initial_params=params)
        if converged(params):
            return params
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))


def lsr_pairwise(n_items, data, alpha=0.0, initial_params=None):
    """Compute the LSR estimate of model parameters.

    This function implements the Luce Spectral Ranking inference algorithm
    [MG15]_ for pairwise comparison data (see :ref:`data-pairwise`).

    The argument ``initial_params`` can be used to iteratively refine an
    existing parameter estimate (see the implementation of
    :func:`~choix.ilsr_pairwise` for an idea on how this works). If it is set
    to `None` (the default), the all-ones vector is used.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise comparison data.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to build the transition rates of the LSR Markov chain.

    Returns
    -------
    params : np.array
        An estimate of model parameters.
    """
    ws, chain = _init_lsr(n_items, alpha, initial_params)
    for winner, loser in data:
        chain[loser, winner] += 1 / (ws[winner] + ws[loser])
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def ilsr_pairwise(n_items, data, alpha=0.0, initial_params=None,
        max_iter=100, tol=1e-8):
    """Compute the ML estimate of model parameters using I-LSR.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given pairwise comparison data (see :ref:`data-pairwise`), using
    the iterative Luce Spectral Ranking algorithm [MG15]_.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise comparison data.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
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
    return _ilsr(n_items, data, alpha, initial_params, max_iter, tol,
            lsr_pairwise)


def lsr_rankings(n_items, data, alpha=0.0, initial_params=None):
    """Compute the LSR estimate of model parameters.

    This function implements the Luce Spectral Ranking inference algorithm
    [MG15]_ for ranking data (see :ref:`data-rankings`).

    The argument ``initial_params`` can be used to iteratively refine an
    existing parameter estimate (see the implementation of
    :func:`~choix.ilsr_rankings` for an idea on how this works). If it is set
    to `None` (the default), the all-ones vector is used.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Ranking data.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to build the transition rates of the LSR Markov chain.

    Returns
    -------
    params : np.array
        An estimate of model parameters.
    """
    ws, chain = _init_lsr(n_items, alpha, initial_params)
    for ranking in data:
        sum_ = ws.take(ranking).sum()
        for i, winner in enumerate(ranking[:-1]):
            val = 1.0 / sum_
            for loser in ranking[i+1:]:
                chain[loser, winner] += val
            sum_ -= ws[winner]
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def ilsr_rankings(n_items, data, alpha=0.0, initial_params=None,
        max_iter=100, tol=1e-8):
    """Compute the ML estimate of model parameters using I-LSR.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given ranking data (see :ref:`data-rankings`), using the
    iterative Luce Spectral Ranking algorithm [MG15]_.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Ranking data.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
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
    return _ilsr(n_items, data, alpha, initial_params, max_iter, tol,
            lsr_rankings)


def lsr_top1(n_items, data, alpha=0.0, initial_params=None):
    """Compute the LSR estimate of model parameters.

    This function implements the Luce Spectral Ranking inference algorithm
    [MG15]_ for top-1 data (see :ref:`data-top1`).

    The argument ``initial_params`` can be used to iteratively refine an
    existing parameter estimate (see the implementation of
    :func:`~choix.ilsr_top1` for an idea on how this works). If it is set to
    `None` (the default), the all-ones vector is used.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Top-1 data.
    alpha : float
        Regularization parameter.
    initial_params : array_like
        Parameters used to build the transition rates of the LSR Markov chain.

    Returns
    -------
    params : np.array
        An estimate of model parameters.
    """
    ws, chain = _init_lsr(n_items, alpha, initial_params)
    for winner, losers in data:
        val = 1 / (ws.take(losers).sum() + ws[winner])
        for loser in losers:
            chain[loser, winner] += val
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def ilsr_top1(n_items, data, alpha=0.0, initial_params=None,
        max_iter=100, tol=1e-8):
    raise RuntimeError("not yet implemented.")
    #return _ilsr(n_items, data, max_iter, tol,
    #        lsr_pairwise, log_likelihood_pairwise)
