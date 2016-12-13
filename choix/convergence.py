import abc
import numpy as np


class ConvergenceTest(metaclass=abc.ABCMeta):

    """Abstract base class for convergence tests.

    Convergence tests should implement a single function, `__call__`, which
    takes a parameter vector and returns a boolean indicating whether or not
    the convergence criterion is met.
    """

    @abc.abstractmethod
    def __call__(self, params, update=True):
        """Test whether convergence criterion is met.

        The parameter `update` controls whether `params` should replace the
        previous parameters (i.e., modify the state of the object).
        """


class NormOfDifferenceTest(ConvergenceTest):

    """Convergence test based on the norm of the difference vector.

    This convergence test computes the difference between two successive
    parameter vectors in log-space, and declares convergence when the norm of
    this difference vector (normalized by the number of items) is below `tol`.
    """

    def __init__(self, tol=1e-8, order=1):
        self._tol = tol
        self._ord = order
        self._prev_thetas = None

    def __call__(self, params, update=True):
        thetas = np.log(params)
        thetas -= np.mean(thetas)
        if self._prev_thetas is None:
            if update:
                self._prev_thetas = thetas
            return False
        dist = np.linalg.norm(self._prev_thetas - thetas, ord=self._ord)
        if update:
            self._prev_thetas = thetas
        return dist <= self._tol * len(thetas)


class ScalarFunctionTest(ConvergenceTest):

    """Convergence test based on a scalar function of the parameters.

    This convergence test computes the values of a scalar function of the
    parameters, and declares convergence when the absolute difference between
    two successive values is below `tol`.

    A typical use case of this class is in conjunction with a log-likelihood
    function.
    """

    def __init__(self, fun, tol=1e-8):
        self._fun = fun
        self._tol = tol
        self._prev_val = None

    def __call__(self, params, update=True):
        val = self._fun(params)
        if self._prev_val is None:
            if update:
                self._prev_val = val
            return False
        dist = abs(val - self._prev_val)
        if update:
            self._prev_val = val
        return dist < self._tol
