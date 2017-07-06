import numpy as np
import pytest

from choix.ep import (
        ep_pairwise, _ep_pairwise, _match_moments_probit, _log_phi)
from math import log, sqrt, pi


DATA = [(3, 2), (2, 1), (1, 0)]

# Expected results if prior variance = 8.0
LOGIT_TRUE_MEAN = np.array(
    [-2.24876774, -0.54278959, 0.5427896, 2.24876774])
LOGIT_TRUE_COV = np.array(
    [[ 5.13442507, 1.78941794, 0.7363867,  0.33977029],
     [ 1.78941794, 3.87821895, 1.59597641, 0.73638669],
     [ 0.7363867,  1.59597641, 3.87821895, 1.78941794],
     [ 0.33977029, 0.73638669, 1.78941794, 5.13442508]])

# Expected results if prior variance = 8.0
PROBIT_TRUE_MEAN = np.array(
    [-2.64311237, -0.71902005, 0.71901978, 2.64311263])
PROBIT_TRUE_COV = np.array(
    [[4.40599359, 1.91559034, 1.05514881, 0.62326726],
     [1.91559034, 3.24296482, 1.78629554, 1.0551493 ],
     [1.05514881, 1.78629554, 3.24296458, 1.91559107],
     [0.62326726, 1.0551493,  1.91559107, 4.40599238]])


def test_ep_pairwise_unknown_model():
    """Unknown method should raise a ValueError."""
    with pytest.raises(ValueError):
        ep_pairwise(8, [], 1.0, model="qwerty")


def test_ep_pairwise_logit():
    """EP should work as expected with the logit model."""
    mean, cov = ep_pairwise(4, DATA, 1/8)
    assert np.allclose(mean, LOGIT_TRUE_MEAN, atol=1e-3)
    assert np.allclose(cov, LOGIT_TRUE_COV, atol=1e-3)


def test_ep_pairwise_probit():
    """EP should work as expected with the probit model."""
    mean, cov = ep_pairwise(4, DATA, 1/8, model="probit")
    assert np.allclose(mean, PROBIT_TRUE_MEAN, atol=1e-3)
    assert np.allclose(cov, PROBIT_TRUE_COV, atol=1e-3)


def test_ep_pairwise_max_iter():
    """Should fail if no convergence after ``max_iter`` iterations."""
    data =[(1, 0), (1, 0)] 
    with pytest.raises(RuntimeError):
        _ep_pairwise(
                2, data, 1.0, _match_moments_probit, max_iter=1,
                initial_state=None)
    assert _ep_pairwise.iterations == 1


def test_ep_pairwise_state():
    """EP should take into account the initial state."""
    data =[(1, 0)]
    tau = np.array([0, 0], dtype=float)
    nu = np.array([0, 0], dtype=float)
    # First run.
    mean1, cov1 = ep_pairwise(2, data, 1.0, initial_state=(tau, nu))
    # Second run, with initial state carried over from first run.
    mean2, cov2 = ep_pairwise(2, data, 1.0, initial_state=(tau, nu))
    assert np.allclose(mean1, mean2)
    assert np.allclose(cov1, cov2)
    assert _ep_pairwise.iterations == 1


def test_log_phi():
    """``_log_phi`` should work as expected."""
    # First case: z close to 0.
    res, dres = _log_phi(0)
    assert np.allclose(res, -log(2))
    assert np.allclose(dres, 2 / sqrt(2 * pi))
    # Second case: z small. Ground truth computed with GPy-1.7.7.
    res, dres = _log_phi(-15)
    assert np.allclose(res, -116.13138484571169)
    assert np.allclose(dres, 15.066086827167823)
    # Third case: z positive, large.
    res, dres = _log_phi(100)
    assert np.allclose(res, 0)
    assert np.allclose(dres, 0)
