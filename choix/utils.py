"""Utilities for ranking-related problems."""
import math
import numpy as np
import random

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


def trunc(arr, a, b):
    """Return the 2 x 2 submatrix by taking elements related to a & b."""
    m = arr.shape[0]
    return arr.take([[a*m + a, a*m + b], [b*m + a, b*m + b]])


def generate_comparisons(model, nb):
    items = range(len(model))
    comparisons = list()
    for x in xrange(nb):
        # Pick the pair uniformly at random.
        a, b = random.sample(items, 2)
        if model.compare(a, b) == a:
            comparisons.append((a, b))
        else:
            comparisons.append((b, a))
    return comparisons


def displacement(ranks, other, avg=False):
    """Compute the rank displacement.

    `ranks` is *not* the ranking; it gives the rank of each item (akin to a
    dictionnary mapping items to their rank.)

    If `avg` is true, normalizes the result by the number of items.
    """
    assert sorted(ranks) == sorted(other)
    total = 0
    for i, j in zip(ranks, other):
        total += abs(i - j)
    return total / float(len(ranks)) if avg else total


class Parameters(object):

    def __init__(self, params):
        self.params = params

    @property
    def ranks(self):
        """Compute the rank of each item."""
        items = range(len(self))
        random.shuffle(items)  # Randomizes things in case of ties.
        ordered = sorted(items, key=lambda x: self.params[x], reverse=True)
        ranks = np.zeros(len(self), dtype=int)
        for rank, item in enumerate(ordered, start=1):
            ranks[item] = rank
        return ranks

    def __len__(self):
        return len(self.params)


class PiParams(Parameters):
    """Utilities to compare parameters."""
    pass


class ThetaParams(Parameters):
    pass
