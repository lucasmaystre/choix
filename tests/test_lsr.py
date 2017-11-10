import numpy as np
import pytest

from choix.lsr import *
from choix.lsr import _ilsr

from unittest.mock import Mock
from tutils import iter_testcases


# Tolerance values for calls to `numpy.allclose`.
ATOL = 1e-4
RTOL = 1e-5


def test_ilsr_tolerance():
    """Tolerance affects the number of iterations."""
    vals = [np.array([-0.5, 0.5]), np.array([-0.3, 0.3]),
            np.array([-0.2, 0.2]), np.array([-0.25, 0.25])]
    lsr = Mock(side_effect=vals)
    est = _ilsr(fun=lsr, params=None, max_iter=100, tol=0.15)
    assert np.array_equal(est, vals[2])
    assert lsr.call_count == 3


def test_ilsr_max_iter():
    """Low `max_iter` raises `RuntimeError`."""
    vals = [np.array([-0.5, 0.5]), np.array([-0.3, 0.3]),
            np.array([-0.2, 0.2]), np.array([-0.25, 0.25])]
    lsr = Mock(side_effect=vals)
    with pytest.raises(RuntimeError):
        _ilsr(fun=lsr, params=None, max_iter=2, tol=0.01)
    assert lsr.call_count == 2


def test_lsr_pairwise():
    """JSON test cases for LSR pairwise."""
    for case in iter_testcases('pairwise'):
        n_items = case["n_items"]
        data = case["data"]
        assert np.allclose(
                case["lsr_est"], lsr_pairwise(n_items, data),
                atol=ATOL, rtol=RTOL)


def test_ilsr_pairwise():
    """JSON test cases for I-LSR pairwise."""
    for case in iter_testcases('pairwise'):
        n_items = case["n_items"]
        data = case["data"]
        assert np.allclose(
                case["ml_est"], ilsr_pairwise(n_items, data),
                atol=ATOL, rtol=RTOL)


def test_lsr_pairwise_dense():
    """JSON test cases for LSR pairwise."""
    for case in iter_testcases('pairwise'):
        n_items = case["n_items"]
        mat = np.zeros((n_items, n_items), dtype=float)
        for a, b in case["data"]:
            mat[a, b] += 1
        assert np.allclose(
                case["lsr_est"], lsr_pairwise_dense(mat),
                atol=ATOL, rtol=RTOL)


def test_ilsr_pairwise_dense():
    """JSON test cases for LSR pairwise."""
    for case in iter_testcases('pairwise'):
        n_items = case["n_items"]
        mat = np.zeros((n_items, n_items), dtype=float)
        for a, b in case["data"]:
            mat[a, b] += 1
        assert np.allclose(
                case["ml_est"], ilsr_pairwise_dense(mat),
                atol=ATOL, rtol=RTOL)


def test_rank_centrality():
    """JSON test cases for Rank Centrality."""
    for case in iter_testcases('pairwise'):
        n_items = case["n_items"]
        data = case["data"]
        assert np.allclose(
                case["rc_est"], rank_centrality(n_items, data),
                atol=ATOL, rtol=RTOL)


def test_lsr_rankings():
    """JSON test cases for LSR rankings."""
    for case in iter_testcases('rankings'):
        n_items = case["n_items"]
        data = case["data"]
        assert np.allclose(
                case["lsr_est"], lsr_rankings(n_items, data),
                atol=ATOL, rtol=RTOL)


def test_ilsr_rankings():
    """JSON test cases for I-LSR rankings."""
    for case in iter_testcases('rankings'):
        n_items = case["n_items"]
        data = case["data"]
        assert np.allclose(
                case["ml_est"], ilsr_rankings(n_items, data),
                atol=ATOL, rtol=RTOL)


def test_lsr_top1():
    """JSON test cases for LSR top1."""
    for case in iter_testcases('top1'):
        n_items = case["n_items"]
        data = case["data"]
        assert np.allclose(
                case["lsr_est"], lsr_top1(n_items, data),
                atol=ATOL, rtol=RTOL)


def test_ilsr_top1():
    """JSON test cases for I-LSR top1."""
    for case in iter_testcases('top1'):
        n_items = case["n_items"]
        data = case["data"]
        assert np.allclose(
                case["ml_est"], ilsr_top1(n_items, data),
                atol=ATOL, rtol=RTOL)
