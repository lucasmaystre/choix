import math
import numpy as np

from scipy.optimize import minimize


def _safe_exp(x):
    x = min(max(x, -500), 500)
    return math.exp(x)


class PairwiseFcts:

    """Optimization-related methods for pairwise comparison data.
    
    This class provides methods to compute the negative log-likelihood (the
    "objective"), its gradient and its Hessian, given model parameters and
    pairwise comparison data.

    The parameters are assumed to be in the log domain.
    """

    def __init__(self, data, penalty):
        self._data = data
        self._penalty = penalty

    def objective(self, params):
        """Compute the negative penalized log-likelihood."""
        val = self._penalty * np.sum(params**2)
        for win, los in self._data:
            val += np.logaddexp(0, -(params[win] - params[los]))
        return val

    def gradient(self, params):
        grad = 2 * self._penalty * params
        for win, los in self._data:
            z = 1 / (1 + _safe_exp(params[win] - params[los]))
            grad[win] += -z
            grad[los] += +z
        return grad

    def hessian(self, params):
        hess = 2 * self._penalty * np.identity(len(params))
        for win, los in self._data:
            z = _safe_exp(params[win] - params[los])
            val =  1 / (1/z + 2 + z)
            hess[(win,los),(los,win)] += -val
            hess[(win,los),(win,los)] += +val
        return hess


def opt_pairwise(n_items, data, penalty=1e-6, method="BFGS",
        initial_params=None, max_iter=None, tol=1e-5):
    """Compute the ML estimate of model parameters using ``scipy.optimize``.

    This function computes the (penalized) maximum-likelihood estimate of model
    parameters given pairwise comparison data (see :ref:`data-pairwise`), using
    optimizers provided by the ``scipy.optimize`` module.

    Parameters
    ----------
    n_items : int
        Number of distinct items.
    data : list of lists
        Pairwise comparison data.
    penalty : float, optional
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
    params : np.array
        The (penalized) ML estimate of model parameters.

    Raises
    ------
    ValueError
        If the method is not "BFGS" or "Newton-CG".
    """
    fcts = PairwiseFcts(data, penalty)
    if initial_params is not None:
        x0 = np.log(initial_params)
        x0 = x0 - np.mean(x0)
    else:
        x0 = np.zeros(n_items)
    if method == "BFGS":
        # `gtol`: Gradient norm must be less than gtol before successful
        # termination [scipy doc].
        res = minimize(fcts.objective, x0, method=method, jac=fcts.gradient,
                options={"gtol": tol, "maxiter": max_iter})
    elif method == "Newton-CG":
        # `xtol`: Average relative error in solution xopt acceptable for
        # convergence [scipy doc].
        res = minimize(fcts.objective, x0, method=method, jac=fcts.gradient,
                hess=fcts.hessian, options={"xtol": tol, "maxiter": max_iter})
    else:
        raise ValueError("method not known")
    # Parameters are in the log domain - reparametrize the model.
    params = np.exp(res.x)
    return params / (params.sum() / n_items)
