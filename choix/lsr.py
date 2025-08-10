"""(Iterative) Luce Spectral Ranking and related inference algorithms."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .convergence import NormOfDifferenceTest
from .typing import PairwiseData, RankingData, Top1Data
from .utils import exp_transform, log_transform, statdist


def _init_lsr(
    n_items: int,
    alpha: float,
    initial_params: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Initialize the LSR Markov chain and the weights."""
    if initial_params is None:
        weights = np.ones(n_items)
    else:
        weights = exp_transform(initial_params)
    chain = alpha * np.ones((n_items, n_items), dtype=float)
    return weights, chain


def _ilsr(
    fun: Callable[[NDArray[np.float64] | None], NDArray[np.float64]],
    params: NDArray[np.float64] | None,
    max_iter: int,
    tol: float,
) -> NDArray[np.float64]:
    """Iteratively refine LSR estimates until convergence.

    Raises
    ------
    RuntimeError
        If the algorithm does not converge after ``max_iter`` iterations.
    """
    converged = NormOfDifferenceTest(tol, order=1)
    for _ in range(max_iter):
        params = fun(params)
        if converged(params):
            return params
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))


def lsr_pairwise(
    n_items: int,
    data: PairwiseData,
    alpha: float = 0.0,
    initial_params: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute the LSR estimate of model parameters.

    This function implements the Luce Spectral Ranking inference algorithm
    [MG15]_ for pairwise-comparison data (see :ref:`data-pairwise`).

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
        Pairwise-comparison data.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to build the transition rates of the LSR Markov chain.

    Returns
    -------
    params : numpy.ndarray
        An estimate of model parameters.
    """
    weights, chain = _init_lsr(n_items, alpha, initial_params)
    for winner, loser in data:
        chain[loser, winner] += 1 / (weights[winner] + weights[loser])
    chain -= np.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))


def ilsr_pairwise(
    n_items: int,
    data: PairwiseData,
    alpha: float = 0.0,
    initial_params: NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """Compute the ML estimate of model parameters using I-LSR.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given pairwise-comparison data (see :ref:`data-pairwise`), using
    the iterative Luce Spectral Ranking algorithm [MG15]_.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise-comparison data.
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
    params : numpy.ndarray
        The ML estimate of model parameters.
    """

    def fun(params: NDArray[np.float64] | None) -> NDArray[np.float64]:
        return lsr_pairwise(n_items=n_items, data=data, alpha=alpha, initial_params=params)

    return _ilsr(fun, initial_params, max_iter, tol)


def lsr_pairwise_dense(
    comp_mat: NDArray[np.float64],
    alpha: float = 0.0,
    initial_params: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute the LSR estimate of model parameters given dense data.

    This function implements the Luce Spectral Ranking inference algorithm
    [MG15]_ for dense pairwise-comparison data.

    The data is described by a pairwise-comparison matrix ``comp_mat`` such
    that ``comp_mat[i,j]`` contains the number of times that item ``i`` wins
    against item ``j``.

    In comparison to :func:`~choix.lsr_pairwise`, this function is particularly
    efficient for dense pairwise-comparison datasets (i.e., containing many
    comparisons for a large fraction of item pairs).

    The argument ``initial_params`` can be used to iteratively refine an
    existing parameter estimate (see the implementation of
    :func:`~choix.ilsr_pairwise` for an idea on how this works). If it is set
    to `None` (the default), the all-ones vector is used.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    comp_mat : np.array
        2D square matrix describing the pairwise-comparison outcomes.
    alpha : float, optional
        Regularization parameter.
    initial_params : array_like, optional
        Parameters used to build the transition rates of the LSR Markov chain.

    Returns
    -------
    params : np.array
        An estimate of model parameters.
    """
    n_items = comp_mat.shape[0]
    ws, chain = _init_lsr(n_items, alpha, initial_params)
    denom = np.tile(ws, (n_items, 1))
    chain += comp_mat.T / (denom + denom.T)
    chain -= np.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))


def ilsr_pairwise_dense(
    comp_mat: NDArray[np.float64],
    alpha: float = 0.0,
    initial_params: NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """Compute the ML estimate of model parameters given dense data.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given dense pairwise-comparison data.

    The data is described by a pairwise-comparison matrix ``comp_mat`` such
    that ``comp_mat[i,j]`` contains the number of times that item ``i`` wins
    against item ``j``.

    In comparison to :func:`~choix.ilsr_pairwise`, this function is
    particularly efficient for dense pairwise-comparison datasets (i.e.,
    containing many comparisons for a large fraction of item pairs).

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    comp_mat : np.array
        2D square matrix describing the pairwise-comparison outcomes.
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
    params : numpy.ndarray
        The ML estimate of model parameters.
    """

    def fun(params: NDArray[np.float64] | None) -> NDArray[np.float64]:
        return lsr_pairwise_dense(comp_mat=comp_mat, alpha=alpha, initial_params=params)

    return _ilsr(fun, initial_params, max_iter, tol)


def rank_centrality(n_items: int, data: PairwiseData, alpha: float = 0.0) -> NDArray[np.float64]:
    """Compute the Rank Centrality estimate of model parameters.

    This function implements Negahban et al.'s Rank Centrality algorithm
    [NOS12]_. The algorithm is similar to :func:`~choix.ilsr_pairwise`, but
    considers the *ratio* of wins for each pair (instead of the total count).

    The transition rates of the Rank Centrality Markov chain are initialized
    with ``alpha``. When ``alpha > 0``, this corresponds to a form of
    regularization (see :ref:`regularization` for details).

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise-comparison data.
    alpha : float, optional
        Regularization parameter.

    Returns
    -------
    params : numpy.ndarray
        An estimate of model parameters.
    """
    _, chain = _init_lsr(n_items, alpha, None)
    for winner, loser in data:
        chain[loser, winner] += 1.0
    # Transform the counts into ratios.
    idx = chain > 0  # Indices (i,j) of non-zero entries.
    chain[idx] = chain[idx] / (chain + chain.T)[idx]
    # Finalize the Markov chain by adding the self-transition rate.
    chain -= np.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))


def lsr_rankings(
    n_items: int,
    data: RankingData,
    alpha: float = 0.0,
    initial_params: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
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
    params : numpy.ndarray
        An estimate of model parameters.
    """
    weights, chain = _init_lsr(n_items, alpha, initial_params)
    for ranking in data:
        sum_ = weights.take(ranking).sum()
        for i, winner in enumerate(ranking[:-1]):
            val = 1.0 / sum_
            for loser in ranking[i + 1 :]:
                chain[loser, winner] += val
            sum_ -= weights[winner]
    chain -= np.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))


def ilsr_rankings(
    n_items: int,
    data: RankingData,
    alpha: float = 0.0,
    initial_params: NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
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
    params : numpy.ndarray
        The ML estimate of model parameters.
    """

    def fun(params: NDArray[np.float64] | None) -> NDArray[np.float64]:
        return lsr_rankings(n_items=n_items, data=data, alpha=alpha, initial_params=params)

    return _ilsr(fun, initial_params, max_iter, tol)


def lsr_top1(
    n_items: int,
    data: Top1Data,
    alpha: float = 0.0,
    initial_params: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
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
    params : numpy.ndarray
        An estimate of model parameters.
    """
    weights, chain = _init_lsr(n_items, alpha, initial_params)
    for winner, losers in data:
        val = 1 / (weights.take(losers).sum() + weights[winner])
        for loser in losers:
            chain[loser, winner] += val
    chain -= np.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))


def ilsr_top1(
    n_items: int,
    data: Top1Data,
    alpha: float = 0.0,
    initial_params: NDArray[np.float64] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
):
    """Compute the ML estimate of model parameters using I-LSR.

    This function computes the maximum-likelihood (ML) estimate of model
    parameters given top-1 data (see :ref:`data-top1`), using the
    iterative Luce Spectral Ranking algorithm [MG15]_.

    The transition rates of the LSR Markov chain are initialized with
    ``alpha``. When ``alpha > 0``, this corresponds to a form of regularization
    (see :ref:`regularization` for details).

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Top-1 data.
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
    params : numpy.ndarray
        The ML estimate of model parameters.
    """

    def fun(params: NDArray[np.float64] | None) -> NDArray[np.float64]:
        return lsr_top1(n_items=n_items, data=data, alpha=alpha, initial_params=params)

    return _ilsr(fun, initial_params, max_iter, tol)
