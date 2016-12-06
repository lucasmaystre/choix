import numpy as np
import pytest

from choix.lsr import *
from choix.lsr import _ilsr

from testutils import data_path, parse_pairwise
from unittest.mock import Mock


def test_ilsr_epsilon():
    val = np.ones(10)
    lsr = Mock(return_value=val)
    ll = Mock(side_effect=[0.5, 1.1, 1.5, 1.7, 1.8])
    est = _ilsr(10, [], alpha=0.0, max_iter=100, eps=0.3,
            lsr_fun=lsr, ll_fun=ll)
    assert np.array_equal(est, val)
    assert lsr.call_count == 4
    assert ll.call_count == 4


def test_ilsr_max_iter():
    val = np.ones(10)
    lsr = Mock(return_value=val)
    ll = Mock(side_effect=[1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError):
        est = _ilsr(10, [], alpha=0.0, max_iter=2, eps=0.5,
                lsr_fun=lsr, ll_fun=ll)


def test_lsr_pairwise():
    data1 = ((0,1), (1,2), (2,0))
    est1 = lsr_pairwise(3, data1)
    assert np.allclose(est1, np.array([1.0, 1.0, 1.0]))
    data2 = ((0,1), (0,1), (1,2), (2,0))
    est2 = lsr_pairwise(3, data2)
    assert np.allclose(est2, np.array([1.2, 0.6, 1.2]))


def test_ilsr_pairwise():
    data1 = ((0,1), (1,2), (2,0))
    est1 = ilsr_pairwise(3, data1)
    assert np.allclose(est1, np.array([1.0, 1.0, 1.0]))
    data2 = ((0,1), (0,1), (1,2), (2,0))
    est2 = ilsr_pairwise(3, data2)
    assert np.allclose(est2, np.array(
            [1.43584141, 0.62040245, 0.94375613]))
