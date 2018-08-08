import networkx as nx
import numpy as np
import pytest

from choix.mm import *
from tutils import iter_testcases
from unittest import mock


RND = np.random.RandomState(42)

# Tolerance values for calls to `numpy.allclose`.
ATOL = 1e-4
RTOL = 1e-5


def test_mm_pairwise():
    """JSON test cases for pairwise data."""
    for case in iter_testcases('pairwise'):
        n_items = case["n_items"]
        data = case["data"]
        for params in (None, RND.randn(n_items)):
            est = mm_pairwise(n_items, data, initial_params=params)
            assert np.allclose(case["ml_est"], est, atol=ATOL, rtol=RTOL)


def test_mm_rankings():
    """JSON test cases for ranking data."""
    for case in iter_testcases('rankings'):
        n_items = case["n_items"]
        data = case["data"]
        for params in (None, RND.randn(n_items)):
            est = mm_rankings(n_items, data, initial_params=params)
            assert np.allclose(case["ml_est"], est, atol=ATOL, rtol=RTOL)


def test_mm_top1():
    """JSON test cases for top-1 data."""
    for case in iter_testcases('top1'):
        n_items = case["n_items"]
        data = case["data"]
        for params in (None, RND.randn(n_items)):
            est = mm_top1(n_items, data, initial_params=params)
            assert np.allclose(case["ml_est"], est, atol=ATOL, rtol=RTOL)


def test_mm_diverges():
    """Should raise an exception if no convergence after ``max_iter``."""
    # MM performs poorly on cycle datasets.
    data = ((0, 1), (1, 2), (2, 3), (3, 0))
    params = np.array([1.0, 0.1, 0.1, 0.1])
    # Pairwise.
    with pytest.raises(RuntimeError):
        mm_pairwise(4, data, initial_params=params, max_iter=10)
    # Ranking.
    with pytest.raises(RuntimeError):
        mm_rankings(4, data, initial_params=params, max_iter=10)
    # Top-1.
    data = ((0, (1,)), (1, (2,)), (2, (3,)), (3, (0,)))
    with pytest.raises(RuntimeError):
        mm_top1(4, data, initial_params=params, max_iter=10)


def test_choicerank_no_networkx():
    with mock.patch.dict('sys.modules', {'networkx': None}):
        with pytest.raises(ImportError):
            choicerank(None, [], [])


def test_choicerank_simple():
    """ChoiceRank should return no-nonsense results on a toy dataset."""
    digraph = nx.DiGraph(incoming_graph_data=[(0, 1), (1, 2), (2, 0)])
    traffic_in = [10, 10, 10]
    traffic_out = [10, 10, 10]
    truth = [0.0, 0.0, 0.0]
    for params in (None, RND.randn(3)):
        est = choicerank(
                digraph, traffic_in, traffic_out, initial_params=params)
        assert np.allclose(truth, est, atol=ATOL, rtol=RTOL)


def test_choicerank_weighted():
    """ChoiceRank should work with weighted networks."""
    digraph = nx.DiGraph()
    digraph.add_weighted_edges_from(((0, 1, 1.0), (0, 2, 2.0)))
    traffic_in = [0, 10, 20]
    traffic_out = [30, 0, 0]
    truth = [0.0, 0.0, 0.0]
    for params in (None, RND.randn(3)):
        est = choicerank(
                digraph, traffic_in, traffic_out, initial_params=params,
                weight='weight')
        assert np.allclose(truth, est, atol=ATOL, rtol=RTOL)


def test_choicerank_complex():
    """Slightly more involved test case for ChoiceRank."""
    # Test data generated using `choicerank.py` from the ChoiceRank's paper
    # supplementary material.
    n_items = 8
    edges = [(0, 5), (1, 3), (1, 4), (2, 1), (2, 4), (2, 6), (2, 7), (3, 2),
            (3, 6), (4, 0), (4, 1), (5, 2), (5, 3), (6, 1), (6, 5), (7, 5)]
    digraph = nx.DiGraph(incoming_graph_data=edges)
    traffic_in = [61, 175, 80, 171, 52, 304, 101, 56]
    traffic_out = [113, 121, 129, 114, 134, 132, 133, 124]
    map_est_1 = [0.117729, 0.29774862, -0.72924662, 0.01879213, -0.70499544,
            0.30602036, -0.04242447, 0.73637643]
    map_est_10 = [0.01371515, 0.18006426, -0.5516032, 0.16807068, -0.54010984,
            0.16228541, 0.01224638, 0.55533115]
    for alpha, truth in ((1.0, map_est_1), (10.0, map_est_10)):
        for params in (None, RND.randn(n_items)):
            est = choicerank(
                    digraph, traffic_in, traffic_out, initial_params=params,
                    alpha=alpha)
            assert np.allclose(truth, est, atol=ATOL, rtol=RTOL)
