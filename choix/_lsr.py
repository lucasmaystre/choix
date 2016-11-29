"""(Iterative) Luce Spectral Ranking inference algorithms."""
import numpy as np

from ._utils import statdist, log_likelihood_rankings


def lsr_rankings(nb_items, rankings, initial_strengths=None):
    """Computes a fast estimate of Plackett--Luce model parameters.
    
    Items are expected to be represented by consecutive integers from `0` to
    `n-1`. A (partial) ranking is defined by a tuple containing the items in
    decreasing order of preference. For example, the tuple
    
        (2, 0, 4)
    corresponds to a ranking where `2` is first, `0` is second, and `4` is
    third.
    The estimate is found using the Luce Spectral Ranking algorithm (LSR).
    The argument `initial_strengths` can be used to iteratively refine an
    existing parameter estimate (see the implementation of `ilsr` for an idea
    on how this works).
    Args:
        nb_items (int): The number of distinct items.
        rankings (List[tuple]): The data (partial rankings).
        initial_strengths (Optional[List]): Strengths used to parametrize the
            transition rates of the LSR Markov chain. If `None`, the strengths
            are assumed to be uniform over the items.
    Returns:
        strengths (List[float]): an estimate of the model parameters given
            the data.
    Raises:
        ValueError: If the rankings do not lead to a strongly connected
            comparison graph.
    """
    if initial_strengths is None:
        ws = np.ones(nb_items)
    else:
        ws = np.asarray(initial_strengths)
    chain = np.zeros((nb_items, nb_items), dtype=float)
    for ranking in rankings:
        sum_ = sum(ws[x] for x in ranking)
        for i, winner in enumerate(ranking[:-1]):
            val = 1.0 / sum_
            for loser in ranking[i+1:]:
                chain[loser, winner] += val
            sum_ -= ws[winner]
    chain -= np.diag(chain.sum(axis=1))
    try:
        return statdist(chain)
    except:
        # Ideally we would like to catch `spl.LinAlgError` only, but there seems
        # to be a bug in scipy, in the code that raises the LinAlgError (!!).
        raise ValueError("the comparison graph is not strongly connected")


def ilsr_rankings(nb_items, rankings, max_iter=100, eps=1e-8):
    """Compute the ML estimate of Plackett--Luce model parameters.
    
    Items are expected to be represented by consecutive integers from `0` to
    `n-1`. A (partial) ranking is defined by a tuple containing the items in
    decreasing order of preference. For example, the tuple
    
        (2, 0, 4)
    corresponds to a ranking where `2` is first, `0` is second, and `4` is
    third.
    The estimate is found using the Iterative Luce Spectral Ranking algorithm
    (I-LSR).
    Args:
        nb_items (int): The number of distinct items.
        rankings (List[tuple]): The data (partial rankings.)
        max_iter (Optional[int]): The maximum number of iterations.
        eps (Optional[float]): Minimum difference between successive
            log-likelihoods to declare convergence.
    Returns:
        strengths (List[float]): the ML estimate of the model parameters given
            the data.
    Raises:
        ValueError: If the rankings do not lead to a strongly connected
            comparison graph.
        RuntimeError: If the algorithm does not converge after `max_iter`
            iterations.
    """
    strengths = np.ones(nb_items)
    prev_loglik = -np.inf
    for _ in range(max_iter):
        strengths = lsr_rankings(nb_items, rankings,
                initial_strengths=strengths)
        loglik = log_likelihood_rankings(rankings, strengths)
        if abs(loglik - prev_loglik) < eps:
            return strengths
        prev_loglik = loglik
    raise RuntimeError("Did not converge after {} iterations".format(max_iter))
