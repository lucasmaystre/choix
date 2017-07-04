import networkx as nx
import numpy as np
import pytest
import scipy.stats as sps

from choix.utils import *
from math import log, e


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


def test_log_likelihood_network():
    digraph = nx.DiGraph()
    digraph.add_weighted_edges_from(((0, 1, 1.0), (0, 2, 2.0)))
    traffic_in = [0, 2, 4]
    traffic_out = [6, 0, 0]
    params = [1.0, 2.0, 3.0]
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
    gen = np.array([[-1, 1, 0, 0], [1, -2, 1, 0], [0, 0, -1, 1], [0, 0, 1, -1]],
            dtype=float)
    dist = np.array([0., 0., 2., 2.])
    assert np.allclose(statdist(gen), dist)


def test_statdist_two_absorbing_classes():
    """
    ``statdist`` should fail when the graph is disconnected or has more than
    one absorbing class.
    """
    # Markov chain with two absorbing classes, (0, 1) and (3, 4).
    gen1 = np.array([[-1, 1, 0, 0, 0], [1, -1, 0, 0, 0], [0, 1, -2, 1, 0],
            [0, 0, 0, -1, 1], [0, 0, 0, 1, -1]], dtype=float)
    with pytest.raises(ValueError):
        x = statdist(gen1)
    # Markov with two disconnected components, (0, 1) and (2, 3).
    gen2 = np.array([[-1, 1, 0, 0], [1, -1, 0, 0], [0, 0, -1, 1],
            [0, 0, 1, -1]], dtype=float)
    with pytest.raises(ValueError):
        x = statdist(gen2)


def test_normcdf():
    """``normcdf`` should return the value of the normal CDF."""
    for x in 3 * RND.randn(10):
        np.allclose(normcdf(x), sps.norm.cdf(x))


def test_normpdf():
    """``normpdf`` should return the value of the normal PDF."""
    for x in 3 * RND.randn(10):
        np.allclose(normpdf(x), sps.norm.pdf(x))


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
    params1 = np.array([1, 1e10, 1e-10, 1e-10, 1e-10])
    x1 = compare((3, 0, 2, 4), params1)
    assert x1 == 0
    x2 = compare((3, 0, 1, 4), params1)
    assert x2 == 1
    params2 = np.ones(10)
    for _ in range(10):
        items = RND.choice(10, size=3, replace=False)
        assert compare(items, params2) in items


def test_compare_rankings():
    """``compare`` should work as expected for rankings."""
    params = np.array([1, 1e10, 1e-10, 1e-10, 1e-10])
    x1 = compare((3, 0), params, rank=True)
    assert np.array_equal(x1, np.array([0, 3]))
    x2 = compare((3, 0, 1), params, rank=True)
    assert np.array_equal(x2, np.array([1, 0, 3]))
