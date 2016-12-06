"""(Iterative) Luce Spectral Ranking inference algorithms."""
import numpy as np

from .utils import (statdist, log_likelihood_pairwise, log_likelihood_rankings,
        log_likelihood_top1)


def _init_lsr(num_items, alpha, initial_params):
    """Initialize the LSR Markov chain and the weights."""
    if initial_params is None:
        ws = np.ones(num_items)
    else:
        ws = np.asarray(initial_params)
    if alpha > 0:
        vec1 = alpha / ws
        vec2 = 1.0 - ws / num_items
        chain = np.outer(vec1, vec2)
    else:
        chain = np.zeros((num_items, num_items), dtype=float)
    return ws, chain


def _ilsr(num_items, data, alpha, max_iter, eps, lsr_fun, ll_fun):
    """Iteratively refine LSR estimates until convergence.

    Raises
    ------
    RuntimeError
        If the algorithm does not converge after `max_iter` iterations.
    """
    params = np.ones(num_items)
    prev_loglik = -np.inf
    for _ in range(max_iter):
        params = lsr_fun(num_items, data, alpha=alpha, initial_params=params)
        loglik = ll_fun(data, params)
        if abs(loglik - prev_loglik) < eps:
            return params
        prev_loglik = loglik
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))


def lsr_pairwise(num_items, data, alpha=0.0, initial_params=None):
    """Compute the LSR estimate of model parameters.

    This function implements the Luce Spectral Ranking inference algorithm [1]_
    for pairwise comparison data (see :ref:`data-pairwise`).

    The argument ``initial_params`` can be used to iteratively refine an
    existing parameter estimate (see the implementation of
    :func:`~choix.ilsr_pairwise` for an idea on how this works). If it is set
    to `None` (the default), the all-ones vector is used.

    Parameters
    ----------
    num_items : int
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

    References
    ----------
    .. [1] L. Maystre, M. Grossglauser, "Fast and Accurate Inference of
       Plackett-Luce Models", NIPS 2015.
    """
    ws, chain = _init_lsr(num_items, alpha, initial_params)
    for winner, loser in data:
        chain[loser, winner] += 1 / (ws[winner] + ws[loser])
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def ilsr_pairwise(num_items, data, alpha=0.0, max_iter=100, eps=1e-8):
    """Compute the ML estimate of model parameters using I-LSR.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given pairwise comparison data (see :ref:`data-pairwise`), using
    the iterative Luce Spectral Ranking algorithm [1]_.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.

    Parameters
    ----------
    num_items : int
        Number of distinct items.
    data : list of lists
        Pairwise comparison data.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    eps : float, optional
        Minimum difference between successive log-likelihoods to declare
        convergence.

    Returns
    -------
    params : np.array
        The ML estimate of model parameters.

    References
    ----------
    .. [1] L. Maystre, M. Grossglauser, "Fast and Accurate Inference of
       Plackett-Luce Models", NIPS 2015.
    """
    return _ilsr(num_items, data, alpha, max_iter, eps,
            lsr_pairwise, log_likelihood_pairwise)


def lsr_rankings(num_items, data, alpha=0.0, initial_params=None):
    """Compute the LSR estimate of model parameters.

    This function implements the Luce Spectral Ranking inference algorithm [1]_
    for ranking data (see :ref:`data-rankings`).

    The argument ``initial_params`` can be used to iteratively refine an
    existing parameter estimate (see the implementation of
    :func:`~choix.ilsr_rankings` for an idea on how this works). If it is set
    to `None` (the default), the all-ones vector is used.

    Parameters
    ----------
    num_items : int
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

    References
    ----------
    .. [1] L. Maystre, M. Grossglauser, "Fast and Accurate Inference of
       Plackett-Luce Models", NIPS 2015.
    """
    ws, chain = _init_lsr(num_items, alpha, initial_params)
    for ranking in data:
        sum_ = ws.take(ranking).sum()
        for i, winner in enumerate(ranking[:-1]):
            val = 1.0 / sum_
            for loser in ranking[i+1:]:
                chain[loser, winner] += val
            sum_ -= ws[winner]
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def ilsr_rankings(num_items, data, max_iter=100, eps=1e-8):
    """Compute the ML estimate of model parameters using I-LSR.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given ranking data (see :ref:`data-rankings`), using the
    iterative Luce Spectral Ranking algorithm [1]_.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under a (peaked) Dirichlet prior. See :ref:`regularization` for
    details.

    Parameters
    ----------
    num_items : int
        Number of distinct items.
    data : list of lists
        Ranking data.
    alpha : float, optional
        Regularization parameter.
    max_iter : int, optional
        Maximum number of iterations allowed.
    eps : float, optional
        Minimum difference between successive log-likelihoods to declare
        convergence.

    Returns
    -------
    params : np.array
        The ML estimate of model parameters.

    References
    ----------
    .. [1] L. Maystre, M. Grossglauser, "Fast and Accurate Inference of
       Plackett-Luce Models", NIPS 2015.
    """
    return _ilsr(num_items, data, max_iter, eps,
            lsr_rankings, log_likelihood_rankings)


def lsr_top1(num_items, data, alpha=0.0, initial_params=None):
    """Compute the LSR estimate of model parameters.

    This function implements the Luce Spectral Ranking inference algorithm [1]_
    for top-1 data (see :ref:`data-top1`).

    The argument ``initial_params`` can be used to iteratively refine an
    existing parameter estimate (see the implementation of
    :func:`~choix.ilsr_top1` for an idea on how this works). If it is set to
    `None` (the default), the all-ones vector is used.

    Parameters
    ----------
    num_items : int
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

    References
    ----------
    .. [1] L. Maystre, M. Grossglauser, "Fast and Accurate Inference of
       Plackett-Luce Models", NIPS 2015.
    """
    ws, chain = _init_lsr(num_items, alpha, initial_params)
    for winner, losers in data:
        val = 1 / (ws.take(losers).sum() + ws[winner])
        for loser in losers:
            chain[loser, winner] += val
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def ilsr_top1(num_items, data, max_iter=100, eps=1e-8):
    raise RuntimeError("not yet implemented.")
    #return _ilsr(num_items, data, max_iter, eps,
    #        lsr_pairwise, log_likelihood_pairwise)
