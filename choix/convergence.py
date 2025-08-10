import abc
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class ConvergenceTest(metaclass=abc.ABCMeta):
    """Abstract base class for convergence tests.

    Convergence tests should implement a single function, `__call__`, which
    takes a parameter vector and returns a boolean indicating whether or not
    the convergence criterion is met.
    """

    @abc.abstractmethod
    def __call__(self, params: NDArray[np.float64], update: bool = True) -> bool:
        """Test whether convergence criterion is met.

        The parameter `update` controls whether `params` should replace the
        previous parameters (i.e., modify the state of the object).
        """


class NormOfDifferenceTest(ConvergenceTest):
    """Convergence test based on the norm of the difference vector.

    This convergence test computes the difference between two successive
    parameter vectors, and declares convergence when the norm of this
    difference vector (normalized by the number of items) is below `tol`.
    """

    def __init__(self, tol: float = 1e-8, order: int = 1):
        self._tol = tol
        self._ord = order
        self._prev_params = None

    def __call__(self, params: NDArray[np.float64], update: bool = True) -> bool:
        params = np.asarray(params) - np.mean(params)
        if self._prev_params is None:
            if update:
                self._prev_params = params
            return False
        dist = np.linalg.norm(self._prev_params - params, ord=self._ord)
        if update:
            self._prev_params = params
        return bool(dist <= self._tol * len(params))


class ScalarFunctionTest(ConvergenceTest):
    """Convergence test based on a scalar function of the parameters.

    This convergence test computes the values of a scalar function of the
    parameters, and declares convergence when the absolute difference between
    two successive values is below `tol`.

    A typical use case of this class is in conjunction with a log-likelihood
    function.
    """

    def __init__(self, fun: Callable[[NDArray[np.float64]], float], tol: float = 1e-8):
        self._fun = fun
        self._tol = tol
        self._prev_val = None

    def __call__(self, params: NDArray[np.float64], update: bool = True) -> bool:
        val = self._fun(params)
        if self._prev_val is None:
            if update:
                self._prev_val = val
            return False
        dist = abs(val - self._prev_val)
        if update:
            self._prev_val = val
        return dist < self._tol
