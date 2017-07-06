import numpy as np
import pytest

from choix.convergence import *
from unittest.mock import Mock


def test_sf_works():
    """Basic test for ScalarFunctionTest."""
    params = np.arange(5)
    seq = [2.0, 1.0, 2.0, 1.5, 1.3]
    fun = Mock()
    # With `tol=0.7`.
    fun.side_effect = seq
    test = ScalarFunctionTest(fun, tol=0.7)
    assert not test(params)
    assert not test(params)
    assert not test(params)
    assert test(params)
    assert test(params)
    # With `tol=0.5`.
    fun.side_effect = seq
    test = ScalarFunctionTest(fun, tol=0.5)
    assert not test(params)
    assert not test(params)
    assert not test(params)
    assert not test(params)
    assert test(params)


def test_nod_works():
    """Basic test for NormOfDifferenceTest."""
    # With `tol=0.7`.
    test = NormOfDifferenceTest(tol=0.7, order=1)
    assert not test(np.exp([-0.5, +0.5]))
    assert not test(np.exp([+0.5, -0.5]))
    assert test(np.exp([0.0, 0.0]))
    assert test(np.exp([-0.2, +0.2]))
    # With `tol=0.4`.
    test = NormOfDifferenceTest(tol=0.4, order=1)
    assert not test(np.exp([-0.5, +0.5]))
    assert not test(np.exp([+0.5, -0.5]))
    assert not test(np.exp([0.0, 0.0]))
    assert test(np.exp([-0.2, +0.2]))


def test_nod_norm():
    """NODTest should respect the order of the norm."""
    # With L1 norm.
    test = NormOfDifferenceTest(tol=0.8, order=1)
    assert not test([-0.5, +0.5])
    assert not test([+0.5, -0.5])
    # With L2 norm.
    test = NormOfDifferenceTest(tol=0.8, order=2)
    assert not test([-0.5, +0.5])
    assert test([+0.5, -0.5])


def test_sf_no_update():
    """SFTest should not update its state when `update=False`."""
    fun = Mock(side_effect=[2., 1.2, 0.9, 0.8])
    test = ScalarFunctionTest(fun, tol=1.0)
    dummy = np.arange(5)
    assert not test(dummy)
    # `2.0 - 1.2 < 1.0` -> True.
    assert test(dummy, update=False)
    # `2.0 - 0.9 > 1.0` -> False.
    assert not test(dummy)
    # `0.9 - 0.8 < 1.0` -> True
    assert test(dummy)


def test_nod_no_update():
    """NODTest should not update its state when `update=False`."""
    test = NormOfDifferenceTest(tol=0.6, order=1)
    assert not test([-1.0, +1.0])
    # Average norm of difference = 0.5.
    assert test([-0.5, +0.5], update=False)
    # Average norm of difference = 0.8.
    assert not test([-0.2, +0.2])
    # Average norm of difference = 0.1.
    assert test([-0.1, +0.1])


def test_nod_normalization():
    """NODTest should normalize the criterion with the # of items."""
    for i in (1, 2, 5, 10):
        test = NormOfDifferenceTest(tol=0.5, order=1)
        assert not test([-1.7, 1.7] * i)
        assert not test([-1.0, 1.0] * i)
        assert test([-0.8, 0.8] * i)
