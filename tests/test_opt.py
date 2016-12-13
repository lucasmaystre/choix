import math
import numpy as np
import pytest

from choix.opt import PairwiseFcts, opt_pairwise
from scipy.optimize import check_grad, approx_fprime


# `8-random` case.
PAIRWISE_DATA = [
    (7, 3), (2, 0), (5, 2), (4, 2), (2, 1),
    (4, 5), (6, 3), (5, 4), (7, 0), (2, 3),
    (4, 0), (0, 4), (6, 5), (3, 2), (3, 4),
    (3, 4), (5, 2), (7, 3), (7, 6), (6, 5),]
# With regularization set to 0.5.
ESTIMATE = [0.50789183, 0.5376469, 0.62405681, 0.797485, 0.62110142,
        0.85111539, 1.60412956, 2.45657308]

RND = np.random.RandomState(42)
EPS = math.sqrt(np.finfo(float).eps)


def test_pairwise_gradient():
    fcts = PairwiseFcts(PAIRWISE_DATA, 0.2)
    for sigma in np.linspace(1, 20, num=10):
        xs = sigma * RND.randn(8)
        val = approx_fprime(xs, fcts.objective, EPS)
        err = check_grad(fcts.objective, fcts.gradient, xs, epsilon=EPS)
        assert abs(err / np.linalg.norm(val)) < 1e-5


def test_pairwise_hessian():
    fcts = PairwiseFcts(PAIRWISE_DATA, 0.2)
    for sigma in np.linspace(1, 20, num=10):
        xs = sigma * RND.randn(8)
        for i in range(8):
            obj = lambda xs: fcts.gradient(xs)[i]
            grad = lambda xs: fcts.hessian(xs)[i]
            val = approx_fprime(xs, obj, EPS)
            err = check_grad(obj, grad, xs, epsilon=EPS)
            assert abs(err / np.linalg.norm(val)) < 1e-5


def test_opt_pairwise():
    for method in ("BFGS", "Newton-CG"):
        for params in (None, np.exp(RND.randn(8))):
            est = opt_pairwise(8, PAIRWISE_DATA, penalty=0.5, method=method,
                    initial_params=params)
            assert np.allclose(est, ESTIMATE)


def test_opt_pairwise_extreme():
    data = ((0,1), (1, 2))
    for method in ("BFGS", "Newton-CG"):
        est = opt_pairwise(3, data, penalty=0.00001, method=method)
        assert est[0] > est[1] > est[2]


def test_opt_pairwise_valuerror():
    with pytest.raises(ValueError):
        est = opt_pairwise(8, PAIRWISE_DATA, penalty=0.5, method="qwerty")
