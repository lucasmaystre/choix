import math
import numpy as np
import pytest

from choix.opt import *
from scipy.optimize import check_grad, approx_fprime
from tutils import iter_testcases


# `8-random` case.
PAIRWISE_DATA = [
    (7, 3), (2, 0), (5, 2), (4, 2), (2, 1),
    (4, 5), (6, 3), (5, 4), (7, 0), (2, 3),
    (4, 0), (0, 4), (6, 5), (3, 2), (3, 4),
    (3, 4), (5, 2), (7, 3), (7, 6), (6, 5),]
# With regularization set to 0.5.
PAIRWISE_ESTIMATE = [-0.51974039, -0.46280878, -0.31376862, -0.06854694,
        -0.3185146, -0.00346107, 0.63032783, 1.05651257]

# Randomly generated case.
TOP1_DATA = [
    (7, (6, 3, 1)), (6, (4, 5)),
    (5, (0, 4, 1)), (0, (3,)), (3, (0, 1)),]
# With regularization set to 0.5.
TOP1_ESTIMATE = [-0.12223311, -0.53980682, 0.0, -0.08808004, -0.40003309,
        0.24075031, 0.30131247, 0.60809028]

RND = np.random.RandomState(42)
EPS = math.sqrt(np.finfo(float).eps)

# Tolerance values for calls to `numpy.allclose`.
ATOL = 1e-4
RTOL = 1e-5


def _test_gradient(n_items, fcts):
    """Helper for testing the gradient of objective functions."""
    for sigma in np.linspace(1, 20, num=10):
        xs = sigma * RND.randn(n_items)
        val = approx_fprime(xs, fcts.objective, EPS)
        err = check_grad(fcts.objective, fcts.gradient, xs, epsilon=EPS)
        assert abs(err / np.linalg.norm(val)) < 1e-5


def _test_hessian(n_items, fcts):
    """Helper for testing the hessian of objective functions."""
    for sigma in np.linspace(1, 20, num=10):
        xs = sigma * RND.randn(n_items)
        for i in range(n_items):
            obj = lambda xs: fcts.gradient(xs)[i]
            grad = lambda xs: fcts.hessian(xs)[i]
            val = approx_fprime(xs, obj, EPS)
            err = check_grad(obj, grad, xs, epsilon=EPS)
            assert abs(err / np.linalg.norm(val)) < 1e-5


def test_pairwise_gradient():
    """Gradient of pairwise-data objective should be correct."""
    fcts = PairwiseFcts(PAIRWISE_DATA, 0.2)
    _test_gradient(8, fcts)


def test_pairwise_hessian():
    """Hessian of pairwise-data objective should be correct."""
    fcts = PairwiseFcts(PAIRWISE_DATA, 0.2)
    _test_hessian(8, fcts)


def test_top1_gradient():
    """Gradient of top1-data objective should be correct."""
    fcts = Top1Fcts(TOP1_DATA, 0.2)
    _test_gradient(8, fcts)


def test_top1_hessian():
    """Hessian of top1-data objective should be correct."""
    fcts = Top1Fcts(TOP1_DATA, 0.2)
    _test_hessian(8, fcts)


def test_opt_pairwise_simple():
    """Simple test where regularization is needed (ML does not exist)."""
    for method in ("BFGS", "Newton-CG"):
        for params in (None, RND.randn(8)):
            est = opt_pairwise(
                    8, PAIRWISE_DATA, alpha=0.5, method=method,
                    initial_params=params)
            assert np.allclose(est, PAIRWISE_ESTIMATE, atol=ATOL, rtol=RTOL)


def test_opt_top1_simple():
    """Simple test where regularization is needed (ML does not exist)."""
    for method in ("BFGS", "Newton-CG"):
        for params in (None, RND.randn(8)):
            est = opt_top1(
                    8, TOP1_DATA, alpha=0.5, method=method,
                    initial_params=params)
            assert np.allclose(est, TOP1_ESTIMATE, atol=ATOL, rtol=RTOL)


def test_opt_pairwise_extreme():
    """MAP estimate takes extreme values (almost divergent)."""
    data = ((0,1), (1, 2))
    for method in ("BFGS", "Newton-CG"):
        est = opt_pairwise(3, data, alpha=0.00001, method=method)
        assert est[0] > est[1] > est[2]


def test_opt_top1_extreme():
    """MAP estimate takes extreme values (almost divergent)."""
    data = ((0, (1,)), (1, (2,)))
    for method in ("BFGS", "Newton-CG"):
        est = opt_top1(3, data, alpha=0.00001, method=method)
        assert est[0] > est[1] > est[2]


def test_opt_unknown_method():
    """Unknown method should raise a ValueError."""
    with pytest.raises(ValueError):
        opt_pairwise(8, [], method="qwerty")
    with pytest.raises(ValueError):
        opt_rankings(8, [], method="qwerty")
    with pytest.raises(ValueError):
        opt_top1(8, [], method="qwerty")


def test_opt_pairwise_json():
    """JSON test cases for pairwise-data ML estimator."""
    for case in iter_testcases('pairwise'):
        n_items = case["n_items"]
        data = case["data"]
        for method in ("BFGS", "Newton-CG"):
            est = opt_pairwise(n_items, data, alpha=0.0, method=method)
            assert np.allclose(case["ml_est"], est, atol=ATOL, rtol=RTOL)


def test_opt_rankings_json():
    """JSON test cases for ranking-data ML estimator."""
    for case in iter_testcases('rankings'):
        n_items = case["n_items"]
        data = case["data"]
        for method in ("BFGS", "Newton-CG"):
            est = opt_rankings(n_items, data, alpha=0.0, method=method)
            assert np.allclose(case["ml_est"], est, atol=ATOL, rtol=RTOL)


def test_opt_top1_json():
    """JSON test cases for top1-data ML estimator."""
    for case in iter_testcases('top1'):
        n_items = case["n_items"]
        data = case["data"]
        for method in ("BFGS", "Newton-CG"):
            est = opt_top1(n_items, data, alpha=0.0, method=method)
            assert np.allclose(case["ml_est"], est, atol=ATOL, rtol=RTOL)
