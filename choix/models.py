import random

from . import utils
from math import exp


class ChoiceModel(object):

    def __init__(self, thetas):
        self.thetas = thetas

    def choose(self, *alts):
        raise RuntimeError("not implemented")

    def compare(self, a, b):
        raise RuntimeError("not implemented")

    def __len__(self):
        return len(self.thetas)


class LogitModel(ChoiceModel):

    def compare(self, a, b):
        prob = 1.0 / (1.0 + exp(self.thetas[b] - self.thetas[a]))
        return a if random.random() < prob else b


class ProbitModel(ChoiceModel):

    def compare(self, a, b):
        prob = utils.cdf(self.thetas[a] - self.thetas[b])
        return a if random.radom() < prob else b
