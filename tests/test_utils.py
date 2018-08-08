import networkx as nx
import numpy as np
import pytest
import scipy.stats as sps

from choix.utils import *
from math import log, e
from scipy.linalg import inv


RND = np.random.RandomState(42)


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
    assert footrule_dist(params1) == 0
    params2 = np.arange(0, -10, -1)
    assert footrule_dist(params2) == 10**2 / 2
    params3 = np.ones(10)
    assert footrule_dist(params3) == 25.0


def test_kendalltau_dist_error():
    params1 = np.arange(3, dtype=float)
    params2 = np.arange(4, dtype=float)
    with pytest.raises(AssertionError):
        kendalltau_dist(params1, params2)


def test_kendalltau_dist_simple_cases():
    params1 = np.array([+1.0, -1.2, +0.0])
    params2 = np.array([+1.5, -0.2, -0.2])
    params3 = np.array([-1.0, +1.2, +0.0])
    for params in (params1, params2, params3):
        assert kendalltau_dist(params, params) == 0.0
    # Tie broken in a "fortunate" way.
    assert kendalltau_dist(params1, params2) == 0.0
    assert kendalltau_dist(params1, params3) == 3.0
    assert kendalltau_dist(params2, params3) == 3.0


def test_kendalltau_dist_default():
    params1 = np.arange(0, 10)
    assert kendalltau_dist(params1) == 0
    params2 = np.arange(0, -10, -1)
    assert kendalltau_dist(params2) == (10 * 9) / 2
    # This is a deceptive case, the ties just happen to be resolved correctly.
    params3 = np.ones(10)
    assert kendalltau_dist(params3) == 0


def test_rmse_error():
    params1 = np.array([+1.0, -1.2, +0.0, -0.3])
    params2 = params1 - 10.0
    assert np.allclose(rmse(params1, params2), 0)
    params3 = params1 + np.array([+0.1, -0.1, -0.1, +0.1])
    assert np.allclose(rmse(params1, params3), 0.1)


def test_rmse_simple_cases():
    params1 = np.arange(3, dtype=float)
    params2 = np.arange(4, dtype=float)
    with pytest.raises(AssertionError):
        rmse(params1, params2)


def test_log_likelihood_pairwise():
    data1 = ((0,1),)
    data2 = ((0,1), (1,0))
    data3 = ((0,1), (1,2), (2,0))
    params1 = np.zeros(3)
    params2 = np.arange(5)
    assert np.allclose(
            log_likelihood_pairwise(data1, params1), -log(2))
    assert np.allclose(
            log_likelihood_pairwise(data2, params1), -2 * log(2))
    assert np.allclose(
            log_likelihood_pairwise(data3, params1), -3 * log(2))
    assert np.allclose(
            log_likelihood_pairwise(data1, params2), -log(1 + e))
    assert np.allclose(
            log_likelihood_pairwise(data2, params2), 1 - 2 * log(1 + e))
    assert np.allclose(
            log_likelihood_pairwise(data3, params2),
            3 - log(1 + e) - log(e + e*e) - log(e*e + 1))


def test_log_likelihood_rankings():
    data = ((0,1,2,3),(1,3,0))
    params1 = np.zeros(10)
    params2 = np.log(np.linspace(1,2, num=4))
    assert np.allclose(
            log_likelihood_rankings(data, params1),
            -log(4) - 2 * (log(3) + log(2)))
    assert np.allclose(
            log_likelihood_rankings(data, params2),
            -5.486092774024455)


def test_log_likelihood_top1():
    data = ((1, (0,2,3)), (3, (1,2)))
    params1 = np.zeros(10)
    params2 = np.log(np.linspace(1,2, num=4))
    assert np.allclose(
            log_likelihood_top1(data, params1), -(log(4) + log(3)))
    assert np.allclose(
            log_likelihood_top1(data, params2), -2.420368128650429)


def test_log_likelihood_network():
    digraph = nx.DiGraph()
    digraph.add_weighted_edges_from(((0, 1, 1.0), (0, 2, 2.0)))
    traffic_in = [0, 2, 4]
    traffic_out = [6, 0, 0]
    params = np.log([1.0, 2.0, 3.0])
    assert np.allclose(
            log_likelihood_network(
                    digraph, traffic_in, traffic_out, params, weight=None),
            2 * log(2 / 5) + 4 * log(3 / 5))
    assert np.allclose(
            log_likelihood_network(
                    digraph, traffic_in, traffic_out, params, weight="weight"),
            2 * log(2 / 8) + 4 * log(3 / 8))


def test_statdist():
    """``statdist`` should return the stationary distribution."""
    gen1 = np.array([[-1, 2/3, 1/3], [1/3, -1, 2/3], [2/3, 1/3, -1]])
    dist1 = np.array([1., 1., 1.])
    assert np.allclose(statdist(gen1), dist1)
    gen2 = np.array([[-2/3, 2/3, 0], [1/3, -1, 2/3], [0, 1/3, -1/3]])
    dist2 = np.array([3/7, 6/7, 12/7])
    assert np.allclose(statdist(gen2), dist2)


def test_statdist_single_absorbing_class():
    """
    ``statdist`` should work when the graph is not strongly connected, but has
    a single absorbing class.
    """
    # Markov chain where states 0 and 1 are transient, and 2 and 3 are
    # absorbing. It is weakly but not strongly connected, and has a single
    # absorbing class.
    gen = np.array(
            [[-1, 1, 0, 0], [1, -2, 1, 0], [0, 0, -1, 1], [0, 0, 1, -1]],
            dtype=float)
    dist = np.array([0., 0., 2., 2.])
    assert np.allclose(statdist(gen), dist)


def test_statdist_two_absorbing_classes():
    """
    ``statdist`` should fail when the graph is disconnected or has more than
    one absorbing class.
    """
    # Markov chain with two absorbing classes, (0, 1) and (3, 4).
    gen1 = np.array(
            [[-1, 1, 0, 0, 0], [1, -1, 0, 0, 0], [0, 1, -2, 1, 0],
            [0, 0, 0, -1, 1], [0, 0, 0, 1, -1]], dtype=float)
    with pytest.raises(ValueError):
        x = statdist(gen1)
    # Markov with two disconnected components, (0, 1) and (2, 3).
    gen2 = np.array(
            [[-1, 1, 0, 0], [1, -1, 0, 0], [0, 0, -1, 1], [0, 0, 1, -1]],
            dtype=float)
    with pytest.raises(ValueError):
        x = statdist(gen2)


def test_softmax():
    """``softmax`` should work as expected."""
    params1 = np.array([0, 0, 0])
    params2 = np.array([1000, 1000, 2000])
    assert np.allclose(softmax(params1), [1/3, 1/3, 1/3])
    assert np.allclose(softmax(params2), [0, 0, 1])


def test_normal_cdf():
    """``normal_cdf`` should return the value of the normal CDF."""
    for x in 3 * RND.randn(10):
        np.allclose(normal_cdf(x), sps.norm.cdf(x))


def test_normal_pdf():
    """``normal_pdf`` should return the value of the normal PDF."""
    for x in 3 * RND.randn(10):
        np.allclose(normal_pdf(x), sps.norm.pdf(x))


def test_inv_posdef():
    """``inv_posdef`` should return the correct inverse."""
    mat = RND.randn(8, 8)
    mat = mat.dot(mat.T)  # Make it positive semi-definite.
    assert np.allclose(inv_posdef(mat), inv(mat))


def test_generate_params():
    """``generate_params`` should work as expected."""
    params1 = generate_params(10)
    assert len(params1) == 10
    params2 = generate_params(10, ordered=True)
    assert params2.tolist() == sorted(params2)


def test_generate_pairwise():
    """``generate_pairwise`` should work as expected."""
    params = np.exp(RND.rand(10))
    for num in RND.choice(20, size=3, replace=False):
        data = generate_pairwise(params, num)
        assert np.array(data).shape == (num, 2)


def test_generate_rankings():
    """``generate_rankings`` should work as expected."""
    n_items = 10
    params = np.exp(RND.rand(n_items))
    for num in RND.choice(20, size=3, replace=False):
        size = 1 + RND.choice(n_items - 1)
        print(params, num, size)
        data = generate_rankings(params, num, size=size)
        assert np.array(data).shape == (num, size)


def test_compare_choice():
    """``compare`` should work as expected for choices."""
    params1 = np.array([0, 100, -100, -100, -100])
    x1 = compare((3, 0, 2, 4), params1)
    assert x1 == 0
    x2 = compare((3, 0, 1, 4), params1)
    assert x2 == 1
    params2 = np.zeros(10)
    for _ in range(10):
        items = RND.choice(10, size=3, replace=False)
        assert compare(items, params2) in items


def test_compare_rankings():
    """``compare`` should work as expected for rankings."""
    params = np.array([0, 100, -100, -100, -100])
    x1 = compare((3, 0), params, rank=True)
    assert np.array_equal(x1, np.array([0, 3]))
    x2 = compare((3, 0, 1), params, rank=True)
    assert np.array_equal(x2, np.array([1, 0, 3]))


def test_probabilities():
    """``probabilities`` should work as expected."""
    params = np.log([1, 2, 3, 4])
    assert np.allclose(probabilities([0, 2, 3], params), [1/8, 3/8, 4/8])
    assert np.allclose(probabilities([1, 0], params), [2/3, 1/3])
