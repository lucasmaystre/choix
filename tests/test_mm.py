import numpy as np
import pytest

from choix.mm import *
from tutils import iter_testcases


RND = np.random.RandomState(42)

# Tolerance values for calls to `numpy.allclose`.
ATOL = 1e-8
RTOL = 1e-3


def test_mm_pairwise():
    """JSON test cases for pairwise data."""
    for case in iter_testcases('pairwise'):
        n_items = case["n_items"]
        data = case["data"]
        for params in (None, np.exp(RND.randn(n_items))):
            est = mm_pairwise(n_items, data, initial_params=params)
            assert np.allclose(case["ml_est"], est, atol=ATOL, rtol=RTOL)


def test_mm_rankings():
    """JSON test cases for ranking data."""
    for case in iter_testcases('rankings'):
        n_items = case["n_items"]
        data = case["data"]
        for params in (None, np.exp(RND.randn(n_items))):
            est = mm_rankings(n_items, data, initial_params=params)
            assert np.allclose(case["ml_est"], est, atol=ATOL, rtol=RTOL)


def test_mm_top1():
    """JSON test cases for top-1 data."""
    for case in iter_testcases('top1'):
        n_items = case["n_items"]
        data = case["data"]
        for params in (None, np.exp(RND.randn(n_items))):
            est = mm_top1(n_items, data, initial_params=params)
            assert np.allclose(case["ml_est"], est, atol=ATOL, rtol=RTOL)


def test_mm_diverges():
    """Should raise an exception if no convergence after ``max_iter``."""
    data = ((0, 1),)
    # Pairwise.
    with pytest.raises(RuntimeError):
        mm_pairwise(2, data, max_iter=10)
    # Ranking.
    with pytest.raises(RuntimeError):
        mm_rankings(2, data, max_iter=10)
    # Top-1.
    data = ((0, (1,)),)
    with pytest.raises(RuntimeError):
        mm_top1(2, data, max_iter=10)
