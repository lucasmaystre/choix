import itertools
import math

from .bayesinf import Logit


MODEL = Logit


def entropy_reduction(i, j, mean, cov):
    """By convention, the new observation is assumed to be "i wins over j"""
    # Moment matching.
    f_var = cov[i,i] + cov[j,j] - 2*cov[i,j]
    f_mean = mean[i] - mean[j]
    logpart, dlogpart, d2logpart = MODEL.match_moments(f_mean, f_var)
    # Computing the reduction in entropy.
    tau = -d2logpart / (1 + d2logpart * f_var)
    diff = 0.5 * math.log1p(f_var * tau)
    # Computing the probability of observing the outcome (almost for free.)
    prob = math.exp(logpart)
    return diff, prob


def select_pair(mean, cov):
    items = range(len(mean))
    best_pair = None
    best_reduc = None
    for i, j in itertools.combinations(items, 2):
        reduc1, prob1 = entropy_reduction(i, j, mean, cov)
        reduc2, prob2 = entropy_reduction(j, i, mean, cov)
        reduc = (prob1 * reduc1) + (prob2 * reduc2)
        if reduc > best_reduc:
            best_pair = (i, j)
            best_reduc = reduc
    return best_pair
