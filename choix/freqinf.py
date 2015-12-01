#!/usr/bin/env python
"""Inference algorithms for Bradley--Terry and Plackett--Luce models.

In general:
- n = number of items / alternatives
- m = number of samples (comparisons, partial or full rankings)
- k = number of items in a partial ranking
"""

from __future__ import division

import argparse
import math
import numpy as np
import numpy.random as npr
import scipy.linalg as spl
import scipy.sparse.linalg as spsl
import statsmodels.api as sm
import statsmodels.base.model
import random

from scipy.misc import logsumexp


LOG_2 = math.log(2.0)
SQRT_2 = math.sqrt(2.0)


###
# Inference algorithms working with pairwise comparisons.

def rc(n, comparisons):
    chain = np.zeros((n, n), dtype=float)
    for winner, loser in comparisons:
        chain[loser, winner] += 1.0
    # Transform the counts into ratios.
    idx = chain > 0  # Indices (i,j) of non-zero entries.
    chain[idx] = chain[idx] / (chain + chain.T)[idx]
    # Finalize the Markov chain by adding the self-transition rate.
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def wlsr_pairwise(n, comparisons, weights=None):
    chain = np.zeros((n, n), dtype=float)
    if weights is not None:
        for winner, loser in comparisons:
            chain[loser, winner] += 1.0 / (weights[winner] + weights[loser])
    else:
        for winner, loser in comparisons:
            chain[loser, winner] += 1.0
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def mm_pairwise(n, comparisons, weights=None):
    """Hunter's minorization-maximization algorithm."""
    if weights is None:
        weights = np.ones(n)
    wts = np.zeros(n, dtype=float)
    denoms = np.zeros(n, dtype=float)
    for winner, loser in comparisons:
        # Each item ranked second to last or better receives 1.0.
        wts[winner] += 1.0
        val = 1.0 / (weights[winner] + weights[loser])
        denoms[winner] += val
        denoms[loser] += val
    res = wts / denoms
    return (n / res.sum()) * res


def logit(n, comparisons):
    mat = np.zeros((len(comparisons), n), dtype=float)
    for i, (winner, loser) in enumerate(comparisons):
        mat[i,winner] = 1.0
        mat[i,loser] = -1.0
    # Constraint on sum of means, see Glickman's note on Zermelo paper.
    add = np.transpose(np.tile(-mat[:,-1], (n-1,1)))
    exog = mat[:,:-1] + add
    endog = np.ones(len(comparisons))
    model = sm.Logit(endog, exog)
    res = model.fit()
    return np.append(res.params, -res.params.sum())


###
# Inference algorithms working with comparisons containing ties.

def wlsr_ties(n, comparisons, ties, weights=None, theta=SQRT_2):
    chain = np.zeros((n, n), dtype=float)
    if weights is None:
        weights = np.ones(n, dtype=float)
    for w, l in comparisons:
        chain[l, w] += 1.0 / (weights[w] + theta * weights[l])
    for a, b in ties:
        coeff = 1.0 / ((theta * weights[a] + weights[b])
                       * (weights[a] + theta * weights[b]))
        chain[a, b] += coeff * weights[a]
        chain[b, a] += coeff * weights[b]
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def mm_ties(n, comparisons, ties, weights=None, theta=SQRT_2):
    if weights is None:
        weights = np.ones(n, dtype=float)
    num = np.zeros(n, dtype=float)
    denom = np.zeros(n, dtype=float)
    for winner, loser in comparisons:
        num[winner] += 1.0
        val = 1.0 / (weights[winner] + theta * weights[loser])
        denom[winner] += val
        denom[loser] += theta * val
    for a, b in ties:
        num[a] += 1.0
        num[b] += 1.0
        denom[a] += (1.0 / (weights[a] + theta * weights[b])
                     + theta / (theta * weights[a] + weights[b]))
        denom[b] += (1.0 / (weights[b] + theta * weights[a])
                     + theta / (theta * weights[b] + weights[a]))
    res = num / denom
    return (n / res.sum()) * res


###
# Inference algorithms working with k-way partial rankings.

def mm_iter(n, rankings, weights=None):
    """Hunter's minorization-maximization algorithm."""
    if weights is None:
        weights = np.ones(n)
    wts = np.zeros(n, dtype=float)
    denoms = np.zeros(n, dtype=float)
    for ranking in rankings:
        # Each item ranked second to last or better receives 1.0.
        wts[list(ranking[:-1])] += 1.0
        sum_weights = sum(weights[x] for x in ranking)
        for idx, i in enumerate(ranking[:-1]):
            val = 1.0 / sum_weights
            for s in ranking[idx:]:
                denoms[s] += val
            sum_weights -= weights[i]
    res = wts / denoms
    return (n / res.sum()) * res


def lsr(n, rankings):
    """Simplified version without weighting."""
    chain = np.zeros((n, n), dtype=float)
    for ranking in rankings:
        for i, winner in enumerate(ranking):
            val = 1.0 / (len(ranking) - i)
            for loser in ranking[i+1:]:
                chain[loser, winner] += val
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


def wlsr(n, rankings, weights=None):
    """Weighted Luce spectral ranking algorithm."""
    if weights is None:
        weights = np.ones(n)
    chain = np.zeros((n, n), dtype=float)
    for ranking in rankings:
        sum_weights = sum(weights[x] for x in ranking)
        for i, winner in enumerate(ranking):
            val = 1.0 / sum_weights
            for loser in ranking[i+1:]:
                chain[loser, winner] += val
            sum_weights -= weights[winner]
    chain -= np.diag(chain.sum(axis=1))
    return statdist(chain)


###
# Inference algorithms working with k-way choices.

def softmax(a):
    tmp = np.exp(a - np.max(a))
    return tmp / np.sum(tmp)


class ChoiceModel(statsmodels.base.model.LikelihoodModel):

    def initialize(self):
        # WARNING: assumes every item was selected at least once.
        self._n = self.endog.max() + 1
        self.start_params = np.zeros(self._n - 1, dtype=float)

    def loglike(self, params):
        params = np.append(params, 0.0)
        ll = 0.0
        for ids, x in zip(self.exog, self.endog):
            ll += params[x] - logsumexp(params[ids])
        return ll

    def score(self, params):
        params = np.append(params, 0.0)
        grad = np.zeros(self._n, dtype=float)
        for ids, x in zip(self.exog, self.endog):
            grad[ids] -= softmax(params[ids])
            grad[x] += 1
        return grad[:-1]

    def hessian(self, params):
        params = np.append(params, 0.0)
        hess = np.zeros((self._n, self._n), dtype=float)
        for ids in self.exog:
            vals = softmax(params[ids])
            hess[np.ix_(ids, ids)] +=  np.outer(vals, vals) - np.diag(vals)
        return hess[:-1,:-1]

    def fit(self, **kwargs):
        res = super(ChoiceModel, self).fit(**kwargs)
        # Add the last parameter back, and zero-mean it for good measure.
        res.params = np.append(res.params, 0)
        res.params -= res.params.mean()
        return res


class RaoKupperModel(statsmodels.base.model.LikelihoodModel):
    """
    This implementation works only if alpha = sqrt(2).

    In this case, a tie is equivalent to one win + one loss.
    """

    def initialize(self):
        # WARNING: assumes every item was selected at least once.
        self._n = self.endog.max() + 1
        self._t = np.log(SQRT_2)
        self.start_params = np.zeros(self._n - 1, dtype=float)

    def loglike(self, params):
        params = np.append(params, 0.0)
        ll = 0.0
        for w, l in self.exog:
            ll += params[w] - logsumexp([params[w], params[l] + self._t])
        return ll

    def score(self, params):
        params = np.append(params, 0.0)
        grad = np.zeros(self._n, dtype=float)
        for w, l in self.exog:
            add = softmax([params[w], params[l] + self._t])
            grad[w] += 1 - add[0]
            grad[l] += -add[1]
        return grad[:-1]

    def hessian(self, params):
        params = np.append(params, 0.0)
        hess = np.zeros((self._n, self._n), dtype=float)
        for w, l in self.exog:
            vals = softmax([params[w], params[l] + self._t])
            hess[np.ix_([w, l], [w, l])] += np.outer(vals, vals) - np.diag(vals)
        return hess[:-1,:-1]

    def fit(self, **kwargs):
        res = super(RaoKupperModel, self).fit(**kwargs)
        # Add the last parameter back, and zero-mean it for good measure.
        res.params = np.append(res.params, 0)
        res.params -= res.params.mean()
        return res


###
# Utilities and other functions.

def gradient_rk(comparisons, ties, weights, theta=SQRT_2):
    """Gradient for the Rao-Kupper model."""
    n = weights.shape[0]
    grad = np.zeros(n, dtype=float)
    for winner, loser in comparisons:
        coeff = 1.0 / (weights[winner] + theta * weights[loser])
        grad[winner] += 1.0 / weights[winner] - coeff
        grad[loser] -= theta* coeff
    for a, b in ties:
        grad[a] += (1.0 / weights[a]
                    - 1.0 / (weights[a] + theta * weights[b])
                    - theta / (theta * weights[a] + weights[b]))
        grad[b] += (1.0 / weights[b]
                    - 1.0 / (weights[b] + theta * weights[a])
                    - theta / (theta * weights[b] + weights[a]))
    return grad


def gradient(rankings, weights):
    n = weights.shape[0]
    grad = np.zeros(n, dtype=float)
    for ranking in rankings:
        for i, winner in enumerate(ranking[:-1]):
            grad[winner] += 1.0 / weights[winner]
            val = 1.0 / sum(weights[x] for x in ranking[i:])
            for alt in ranking[i:]:
                grad[alt] -= val
    return grad


def loglike(rankings, weights):
    res = 0
    for ranking in rankings:
        winner = ranking[0]
        res += math.log(weights[winner])
        res -= math.log(sum(weights[x] for x in ranking))
    return res


def fullbreak(rankings):
    """Break partial rankings into all pairwise comparisons."""
    comparisons = list()
    for ranking in rankings:
        for i, winner in enumerate(ranking):
            for loser in ranking[i+1:]:
                comparisons.append((winner, loser))
    return comparisons


def statdist(generator, method="kernel"):
    """Compute the stationary distribution of a Markov chain.

    The Markov chain is descibed by its infinitesimal generator matrix. It
    needs to be irreducible, but does not need to be aperiodic. Computing the
    stationary distribution can be done with one of the following methods:

    - `kernel`: directly computes the left null space (co-kernel) the generator
      matrix using its LU-decomposition.
    - `eigenval`: finds the leading left eigenvector of an equivalent
      discrete-time MC using `scipy.sparse.linalg.eigs`.
    - `power`: finds the leading left eigenvector of an equivalent
      discrete-time MC using power iterations.
    """
    n = generator.shape[0]
    if method == "kernel":
        # `lu` contains U on the upper triangle, including the diagonal.
        # TODO: this raises a warning when generator is singular (which it, by
        # construction, is! I could add:
        #
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings('ignore')
        #
        # But i don't know what the performance implications are.
        lu, piv = spl.lu_factor(generator.T, check_finite=False)
        # The last row contains 0's only.
        left = lu[:-1,:-1]
        right = -lu[:-1,-1]
        # Solves system `left * x = right`. Assumes that `left` is
        # upper-triangular (ignores lower triangle.)
        res = spl.solve_triangular(left, right, check_finite=False)
        res = np.append(res, 1.0)
        return (n / res.sum()) * res
    if method == "eigenval":
        # DTMC is a row-stochastic matrix.
        eps = 1.0 / np.max(np.abs(generator))
        mat = np.eye(n) + eps*generator
        # Find the leading left eigenvector.
        vals, vecs = spsl.eigs(mat.T, k=1)
        res = np.real(vecs[:,0])
        return (n / res.sum()) * res
    else:
        raise RuntimeError("not (yet?) implemented")


def main():
    pass


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dummy')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main()
