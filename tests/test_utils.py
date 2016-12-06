import numpy as np
import pytest

from choix.utils import *
from math import log, e


def test_footrule_dist_error():
    params1 = np.arange(3, dtype=float)
    params2 = np.arange(4, dtype=float)
    with pytest.raises(AssertionError):
        footrule_dist(params1, params2)


def test_footrule_dist_simple_cases():
    params1 = np.array([+1.0, -1.2, +0.0])
    params2 = np.array([+1.5, -0.2, -0.2])
    params3 = np.array([-1.0, +1.2, +0.0])
    for params in (params1, params2, params3):
        assert footrule_dist(params, params) == 0.0
    assert footrule_dist(params1, params2) == 1.0
    assert footrule_dist(params1, params3) == 4.0


def test_footrule_dist_default():
    params1 = np.arange(0, 10)
    assert footrule_dist(params1) == 10**2 / 2
    params2 = np.arange(0, -10, -1)
    assert footrule_dist(params2) == 0
    params3 = np.ones(10)
    assert footrule_dist(params3) == 25.0


def test_log_likelihood_pairwise():
    data1 = ((0,1),)
    data2 = ((0,1), (1,0))
    data3 = ((0,1), (1,2), (2,0))
    params1 = np.ones(3)
    params2 = np.exp(np.arange(5))
    assert np.allclose(log_likelihood_pairwise(data1, params1),
            -log(2))
    assert np.allclose(log_likelihood_pairwise(data2, params1),
            -2 * log(2))
    assert np.allclose(log_likelihood_pairwise(data3, params1),
            -3 * log(2))
    assert np.allclose(log_likelihood_pairwise(data1, params2),
            -log(1 + e))
    assert np.allclose(log_likelihood_pairwise(data2, params2),
            1 - 2 * log(1 + e))
    assert np.allclose(log_likelihood_pairwise(data3, params2),
            3 - log(1 + e) - log(e + e*e) - log(e*e + 1))


def test_log_likelihood_rankings():
    data = ((0,1,2,3),(1,3,0))
    params1 = e * np.ones(10)
    params2 = np.linspace(1,2, num=4)
    assert np.allclose(log_likelihood_rankings(data, params1),
            -log(4) - 2 * (log(3) + log(2)))
    assert np.allclose(log_likelihood_rankings(data, params2),
            -5.486092774024455)


def test_log_likelihood_top1():
    data = ((1, (0,2,3)), (3, (1,2)))
    params1 = e * np.ones(10)
    params2 = np.linspace(1,2, num=4)
    assert np.allclose(log_likelihood_top1(data, params1),
            -(log(4) + log(3)))
    assert np.allclose(log_likelihood_top1(data, params2),
            -2.420368128650429)
