import numpy as np
import pytest

from choix.mm import *
from tutils import iter_testcases


RND = np.random.RandomState(42)


def test_mm_pairwise():
    for case in iter_testcases('pairwise'):
        num_items = case["num_items"]
        data = case["data"]
        for params in (None, np.exp(RND.randn(num_items))):
            assert np.allclose(case["ml_est"],
                    mm_pairwise(num_items, data, initial_params=params))


def test_mm_pairwise_diverges():
    """Should raise an exception if no convergence after ``max_iter``."""
    data = ((0, 1),)
    with pytest.raises(RuntimeError):
        mm_pairwise(2, data, max_iter=10)
