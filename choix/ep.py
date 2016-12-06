import functools
import numpy as np
import numpy.random as nprand

from numpy.linalg import norm
from math import exp, log, pi, sqrt  # Faster than numpy equivalents.
from numbers import Real
from scipy.misc import logsumexp

from .utils import normcdf, inv_pd, SQRT2, SQRT2PI


# EP-related settings.
MAX_ITERATIONS = 100
THRESHOLD = 1e-4

MAT_ONE = np.array([[1.0, -1.0], [-1.0, 1.0]])
MAT_ONE_FLAT = MAT_ONE.ravel()


# Some magic constants for a stable computation of logphi(z).
CS = [
  0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802,
  0.00556964649138, 0.00125993961762116, -0.01621575378835404,
  0.02629651521057465, -0.001829764677455021, 2*(1-pi/3), (4-pi)/3, 1, 1,]
RS = [
  1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441,
  7.409740605964741794425, 2.9788656263939928886,]
QS = [
  2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034,
  17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677,]


def ep_pairwise(nb_items, data, prior, model="logit",
        max_iter=MAX_ITERATIONS, state=None):
    if model == "logit":
        match_moments = _match_moments_logit
    elif model == "probit":
        match_moments = _match_moments_probit
    else:
        raise ValueError("unknown model '{}'".format(model))
    return _ep_pairwise(nb_items, data, prior, match_moments,
            max_iter=MAX_ITERATIONS, state=state)


def _logphi(z):
    # Adapted from the GPML function `logphi.m`.
    if z * z < 0.0492:
        # First case: z close to zero.
        coef = -z / SQRT2PI
        val = functools.reduce(lambda acc, c: coef * (c + acc), CS, 0)
        res = -2 * val - log(2)
        dres = exp(-(z * z) / 2 - res) / SQRT2PI
    elif z < -11.3137:
        # Second case: z very small.
        num = functools.reduce(lambda acc, r: -z * acc / SQRT2 + r, RS,
                     0.5641895835477550741)
        den = functools.reduce(lambda acc, q: -z * acc / SQRT2 + q, QS, 1.0)
        res = log(num / (2 * den)) - (z * z) / 2
        dres = abs(den / num) * sqrt(2.0 / pi)
    else:
        res = log(normcdf(z))
        dres = exp(-(z * z) / 2 - res) / SQRT2PI
    return res, dres


def _match_moments_logit(mean_cav, cov_cav):
    # Adapted from the GPML function `likLogistic.m`.
    # First use a scale mixture.
    lambdas = sqrt(2) * np.array([0.44, 0.41, 0.40, 0.39, 0.36]);
    cs = np.array([
      1.146480988574439e+02,
      -1.508871030070582e+03,
      2.676085036831241e+03,
      -1.356294962039222e+03,
      7.543285642111850e+01
    ])
    arr1, arr2, arr3 = np.zeros(5), np.zeros(5), np.zeros(5)
    for i, x in enumerate(lambdas):
        arr1[i], arr2[i], arr3[i] = _match_moments_probit(x * mean_cav,
                                                          x*x * cov_cav)
    logpart1 = logsumexp(arr1, b=cs)
    dlogpart1 = (np.dot(np.exp(arr1) * arr2, cs * lambdas)
                 / np.dot(np.exp(arr1), cs))
    d2logpart1 = (np.dot(np.exp(arr1) * (arr2 * arr2 + arr3),
                         cs * lambdas * lambdas)
                  / np.dot(np.exp(arr1), cs)) - (dlogpart1 * dlogpart1)
    # Tail decays linearly in the log domain (and not quadratically.)
    exponent = -10.0 * (abs(mean_cav) - (196.0 / 200.0) * cov_cav - 4.0) 
    if exponent < 500:
        lambd = 1.0 / (1.0 + exp(exponent))
        logpart2 = min(cov_cav / 2.0 - abs(mean_cav), -0.1)
        dlogpart2 = 1.0
        if mean_cav > 0:
            logpart2 = log(1 - exp(logpart2))
            dlogpart2 = 0.0
        d2logpart2 = 0.0
    else:
        lambd, logpart2, dlogpart2, d2logpart2 = 0.0, 0.0, 0.0, 0.0
    logpart = (1 - lambd) * logpart1 + lambd * logpart2
    dlogpart = (1 - lambd) * dlogpart1 + lambd * dlogpart2
    d2logpart = (1 - lambd) * d2logpart1 + lambd * d2logpart2
    return logpart, dlogpart, d2logpart


def _match_moments_probit(mean_cav, cov_cav):
    # Adapted from the GPML function `likErf.m`.
    z = mean_cav / sqrt(1 + cov_cav)
    logpart, val = _logphi(z)
    dlogpart = val / sqrt(1 + cov_cav)  # 1st derivative w.r.t. mean.
    d2logpart = -val * (z + val) / (1 + cov_cav)
    return logpart, dlogpart, d2logpart


def _ep_pairwise(n, comparisons, prior, match_moments, max_iter=MAX_ITERATIONS, state=None):
    # Static variable that allows to check the # of iterations after the call.
    _ep_pairwise.iterations = 0
    m = len(comparisons)
    if isinstance(prior, Real):
        # Shortcut if the prior is isotropic.
        prior = prior * np.eye(n)
    prior_inv = inv_pd(prior)
    if state is None:
        # Initially, mean and covariance come from the prior.
        mean = np.zeros(n)
        cov = np.array(prior, copy=True)
        # Initialize the natural params in the function space.
        tau = np.zeros(m)
        nu = np.zeros(m)
        # Initialize the natural params in the space of thetas.
        prec = np.zeros((n, n))
        xs = np.zeros(n)
    else:
        tau, nu = state
        mean, cov, xs, prec = _init_ws(n, comparisons, prior_inv, tau, nu)
    for _ in range(max_iter):
        _ep_pairwise.iterations += 1
        # Keep a copy of the old parameters for convergence testing.
        tau_old = np.array(tau, copy=True)
        nu_old = np.array(nu, copy=True)
        for i in nprand.permutation(m):
            a, b = comparisons[i]
            # Update mean and variance in function space.
            f_var = cov[a,a] + cov[b,b] - 2 * cov[a,b]
            f_mean = mean[a] - mean[b]
            # Cavity distribution.
            tau_tot = 1.0 / f_var
            nu_tot = tau_tot * f_mean
            tau_cav = tau_tot - tau[i]
            nu_cav = nu_tot - nu[i]
            cov_cav = 1.0 / tau_cav
            mean_cav = cov_cav * nu_cav
            # Moment matching.
            logpart, dlogpart, d2logpart = match_moments(mean_cav, cov_cav)
            # Update factor params in the function space.
            tau[i] = -d2logpart / (1 + d2logpart / tau_cav)
            delta_tau = tau[i] - tau_old[i]
            nu[i] = ((dlogpart - (nu_cav / tau_cav) * d2logpart)
                     / (1 + d2logpart / tau_cav))
            delta_nu = nu[i] - nu_old[i]
            # Update factor params in the weight space.
            prec[(a, a, b, b), (a, b, a, b)] += delta_tau * MAT_ONE_FLAT
            xs[a] += delta_nu
            xs[b] -= delta_nu
            # Update mean and covariance.
            if abs(delta_tau) > 0:
                phi = -1.0 / ((1.0 / delta_tau) + f_var) * MAT_ONE
                upd_mat = cov.take([a, b], axis=0)
                cov = cov + upd_mat.T.dot(phi).dot(upd_mat)
            mean = cov.dot(xs)
        # Recompute the global parameters for stability.
        cov = inv_pd(prior_inv + prec)
        mean = cov.dot(xs)
        if _converged((tau, nu), (tau_old, nu_old)):
            return mean, cov
    raise RuntimeError(
            'EP did not converge after {} iterations'.format(max_iter))


def _init_ws(n, comparisons, prior_inv, tau, nu):
    """Initialize parameters in the weight space."""
    prec = np.zeros((n, n))
    xs = np.zeros(n)
    for i, (a, b) in enumerate(comparisons):
        prec[(a, a, b, b), (a, b, a, b)] += tau[i] * MAT_ONE_FLAT 
        xs[a] += nu[i]
        xs[b] -= nu[i]
    cov = inv_pd(prior_inv + prec)
    mean = cov.dot(xs)
    return mean, cov, xs , prec


def _converged(new, old, threshold=THRESHOLD):
    for param_new, param_old in zip(new, old):
        if norm(param_new - param_old, ord=np.inf) > threshold:
            return False
    return True
