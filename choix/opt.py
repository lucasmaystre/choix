from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .typing import PairwiseData, RankingData, Top1Data
from .utils import softmax


class PairwiseFcts:
    """Optimization-related methods for pairwise-comparison data.

    This class provides methods to compute the negative log-likelihood (the
    "objective"), its gradient and its Hessian, given model parameters and
    pairwise-comparison data.

    Repeated comparisons are aggregated into unique ``(winner, loser)`` pairs
    with counts, and all computations are vectorized over these pairs. The cost
    of each method is therefore linear in the number of distinct pairs (plus the
    number of items), independent of how many times each pair was observed.
    """

    def __init__(self, data: PairwiseData, penalty: float):
        pairs = np.asarray(data, dtype=np.intp).reshape(-1, 2)
        pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        self._win = pairs[:, 0]
        self._los = pairs[:, 1]
        self._counts = counts.astype(np.float64)
        self._penalty = penalty

    def _diffs(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        return params[self._win] - params[self._los]

    def _scatter(self, n_items: int, vals: NDArray[np.float64]) -> NDArray[np.float64]:
        """Add `vals` to the winner of each pair and subtract it from the loser."""
        return np.bincount(self._win, weights=vals, minlength=n_items) - np.bincount(
            self._los, weights=vals, minlength=n_items
        )

    def objective(self, params: NDArray[np.float64]) -> float:
        """Compute the negative penalized log-likelihood."""
        val = self._penalty * np.sum(params**2)
        val += self._counts @ np.logaddexp(0, -self._diffs(params))
        return float(val)

    def gradient(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        # z = P(loser beats winner) = sigmoid(-(params[win] - params[los])).
        z = expit(-self._diffs(params))
        return 2 * self._penalty * params - self._scatter(len(params), self._counts * z)

    def _curvature(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        # Second derivative of each pair's term w.r.t. its difference: p(1-p).
        p = expit(self._diffs(params))
        return self._counts * p * (1 - p)

    def hessian(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        n_items = len(params)
        vals = self._curvature(params)
        hess = 2 * self._penalty * np.identity(n_items)
        hess[np.diag_indices(n_items)] += np.bincount(
            self._win, weights=vals, minlength=n_items
        ) + np.bincount(self._los, weights=vals, minlength=n_items)
        np.add.at(hess, (self._win, self._los), -vals)
        np.add.at(hess, (self._los, self._win), -vals)
        return hess

    def hessp(self, params: NDArray[np.float64], vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the product of the Hessian at `params` with `vec`.

        Unlike :meth:`hessian`, this never materializes the dense Hessian.
        """
        vals = self._curvature(params) * self._diffs(vec)
        return 2 * self._penalty * vec + self._scatter(len(params), vals)


class Top1Fcts:
    """Optimization-related methods for top-1 data.

    This class provides methods to compute the negative log-likelihood (the
    "objective"), its gradient and its Hessian, given model parameters and
    top-1 data.

    The class also provides an alternative constructor for ranking data.
    """

    def __init__(self, data: Top1Data, penalty: float):
        self._data = data
        self._penalty = penalty

    @classmethod
    def from_rankings(cls, data: RankingData, penalty: float) -> "Top1Fcts":
        """Alternative constructor for ranking data."""
        top1 = list()
        for ranking in data:
            for i, winner in enumerate(ranking[:-1]):
                top1.append((winner, ranking[i + 1 :]))
        return cls(top1, penalty)

    def objective(self, params: NDArray[np.float64]) -> float:
        """Compute the negative penalized log-likelihood."""
        val = self._penalty * np.sum(params**2)
        for winner, losers in self._data:
            idx = np.append(winner, losers)
            val += logsumexp(params.take(idx) - params[winner])  # pyright: ignore[reportOperatorIssue]
        return val

    def gradient(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        grad = 2 * self._penalty * params
        for winner, losers in self._data:
            idx = np.append(winner, losers)
            zs = softmax(params.take(idx))
            grad[idx] += zs
            grad[winner] += -1
        return grad

    def hessian(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        hess = 2 * self._penalty * np.identity(len(params))
        for winner, losers in self._data:
            idx = np.append(winner, losers)
            zs = softmax(params.take(idx))
            hess[np.ix_(idx, idx)] += -np.outer(zs, zs)
            hess[idx, idx] += zs
        return hess


def _opt(
    n_items: int,
    fcts: PairwiseFcts | Top1Fcts,
    method: Literal["BFGS", "Newton-CG"],
    initial_params: NDArray[np.float64] | None,
    max_iter: int | None,
    tol: float,
) -> NDArray[np.float64]:
    if initial_params is not None:
        x0 = initial_params
    else:
        x0 = np.zeros(n_items)
    if method == "Newton-CG":
        # `xtol`: Average relative error in solution xopt acceptable for
        # convergence [scipy doc].
        if isinstance(fcts, PairwiseFcts):
            # Hessian-vector products avoid building the dense n x n Hessian.
            hess_kwargs = {"hessp": fcts.hessp}
        else:
            hess_kwargs = {"hess": fcts.hessian}
        res = minimize(
            fcts.objective,
            x0,
            method=method,
            jac=fcts.gradient,
            options={"xtol": tol, "maxiter": max_iter},
            **hess_kwargs,
        )
    elif method == "BFGS":
        # `gtol`: Gradient norm must be less than gtol before successful
        # termination [scipy doc].
        res = minimize(
            fcts.objective,
            x0,
            method=method,
            jac=fcts.gradient,
            options={"gtol": tol, "maxiter": max_iter},
        )
    else:
        raise ValueError("method not known")
    return res.x


def opt_pairwise(
    n_items: int,
    data: PairwiseData,
    alpha: float = 1e-6,
    method: Literal["BFGS", "Newton-CG"] = "Newton-CG",
    initial_params: NDArray[np.float64] | None = None,
    max_iter: int | None = None,
    tol: float = 1e-5,
):
    """Compute the ML estimate of model parameters using ``scipy.optimize``.

    This function computes the maximum-likelihood estimate of model parameters
    given pairwise-comparison data (see :ref:`data-pairwise`), using optimizers
    provided by the ``scipy.optimize`` module.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under an isotropic Gaussian prior with variance ``1 / alpha``. See
    :ref:`regularization` for details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise-comparison data.
    alpha : float, optional
        Regularization strength.
    method : str, optional
        Optimization method. Either "BFGS" or "Newton-CG".
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Tolerance for termination (method-specific).

    Returns
    -------
    params : numpy.ndarray
        The (penalized) ML estimate of model parameters.

    Raises
    ------
    ValueError
        If the method is not "BFGS" or "Newton-CG".
    """
    fcts = PairwiseFcts(data, alpha)
    return _opt(n_items, fcts, method, initial_params, max_iter, tol)


def opt_rankings(
    n_items: int,
    data: RankingData,
    alpha: float = 1e-6,
    method: Literal["BFGS", "Newton-CG"] = "Newton-CG",
    initial_params: NDArray[np.float64] | None = None,
    max_iter: int | None = None,
    tol: float = 1e-5,
):
    """Compute the ML estimate of model parameters using ``scipy.optimize``.

    This function computes the maximum-likelihood estimate of model parameters
    given ranking data (see :ref:`data-rankings`), using optimizers provided by
    the ``scipy.optimize`` module.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under an isotropic Gaussian prior with variance ``1 / alpha``. See
    :ref:`regularization` for details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Ranking data.
    alpha : float, optional
        Regularization strength.
    method : str, optional
        Optimization method. Either "BFGS" or "Newton-CG".
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Tolerance for termination (method-specific).

    Returns
    -------
    params : numpy.ndarray
        The (penalized) ML estimate of model parameters.

    Raises
    ------
    ValueError
        If the method is not "BFGS" or "Newton-CG".
    """
    fcts = Top1Fcts.from_rankings(data, alpha)
    return _opt(n_items, fcts, method, initial_params, max_iter, tol)


def opt_top1(
    n_items: int,
    data: Top1Data,
    alpha: float = 1e-6,
    method: Literal["BFGS", "Newton-CG"] = "Newton-CG",
    initial_params: NDArray[np.float64] | None = None,
    max_iter: int | None = None,
    tol: float = 1e-5,
):
    """Compute the ML estimate of model parameters using ``scipy.optimize``.

    This function computes the maximum-likelihood estimate of model parameters
    given top-1 data (see :ref:`data-top1`), using optimizers provided by the
    ``scipy.optimize`` module.

    If ``alpha > 0``, the function returns the maximum a-posteriori (MAP)
    estimate under an isotropic Gaussian prior with variance ``1 / alpha``. See
    :ref:`regularization` for details.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Top-1 data.
    alpha : float, optional
        Regularization strength.
    method : str, optional
        Optimization method. Either "BFGS" or "Newton-CG".
    initial_params : array_like, optional
        Parameters used to initialize the iterative procedure.
    max_iter : int, optional
        Maximum number of iterations allowed.
    tol : float, optional
        Tolerance for termination (method-specific).

    Returns
    -------
    params : numpy.ndarray
        The (penalized) ML estimate of model parameters.

    Raises
    ------
    ValueError
        If the method is not "BFGS" or "Newton-CG".
    """
    fcts = Top1Fcts(data, alpha)
    return _opt(n_items, fcts, method, initial_params, max_iter, tol)
