import numpy as np
import pytest

from choix.lsr import *
from choix.lsr import _ilsr

from unittest.mock import Mock
from tutils import iter_testcases


def test_ilsr_tolerance():
    vals = [np.exp([-0.5, 0.5]), np.exp([-0.3, 0.3]),
            np.exp([-0.2, 0.2]), np.exp([-0.25, 0.25])]
    lsr = Mock(side_effect=vals)
    est = _ilsr(2, [], alpha=0.0, params=None, max_iter=100, tol=0.15,
            lsr_fun=lsr)
    assert np.array_equal(est, vals[2])
    assert lsr.call_count == 3


def test_ilsr_max_iter():
    vals = [np.exp([-0.5, 0.5]), np.exp([-0.3, 0.3]),
            np.exp([-0.2, 0.2]), np.exp([-0.25, 0.25])]
    lsr = Mock(side_effect=vals)
    with pytest.raises(RuntimeError):
        _ilsr(2, [], alpha=0.0, params=None, max_iter=2, tol=0.01, lsr_fun=lsr)
    assert lsr.call_count == 2


def test_lsr_pairwise():
    data1 = ((0,1), (1,2), (2,0))
    est1 = lsr_pairwise(3, data1)
    assert np.allclose(est1, np.array([1.0, 1.0, 1.0]))
    data2 = ((0,1), (0,1), (1,2), (2,0))
    est2 = lsr_pairwise(3, data2)
    assert np.allclose(est2, np.array([1.2, 0.6, 1.2]))


def test_ilsr_pairwise():
    for case in iter_testcases('pairwise'):
        n_items = case["n_items"]
        data = case["data"]
        assert np.allclose(case["ml_est"], ilsr_pairwise(n_items, data))
