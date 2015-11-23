"""Utilities for ranking-related problems."""
import math
import numpy as np

from scipy.linalg import solve_triangular


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


def normcdf(x):
    """Normal cumulative density function."""
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


class Parameters(object):
    pass


class PiParams(Parameters):
    pass


class ThetaParams(Parameters):
    pass
