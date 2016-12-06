# choix

[![Build Status](https://travis-ci.org/lucasmaystre/choix.svg?branch=master)](https://travis-ci.org/lucasmaystre/choix)
[![codecov](https://codecov.io/gh/lucasmaystre/choix/branch/master/graph/badge.svg)](https://codecov.io/gh/lucasmaystre/choix)
[![Documentation Status](https://readthedocs.org/projects/choix/badge/?version=latest)](http://choix.lum.li/en/latest/?badge=latest)

`choix` is a Python library that provides inference algorithms for models based
on Luce's choice axiom. These (probabilistic) models can be used to explain and
predict outcomes of comparisons between items.

- **Pairwise comparisons**: when the data consists of comparisons between two
  items, the model variant is usually referred to as the *Bradley-Terry* model.
  It is closely related to the Elo rating system used to rank chess players.
- **Partial rankings**: when the data consists of rankings over (a subset of)
  the items, the model variant is usually referred to as the *Plackett-Luce*
  model.
- **Top-1 lists**: another variation of the model arises when the data consists
  of discrete choices, i.e., we observe the selection of one item out of a
  subset of items.

`choix` makes it easy to infer model parameters from these different types of
data, using a variety of algorithms:

- Luce Spectral Ranking
- Minorization-Maximization
- Rank Centrality
- GMM using rank breaking
- Approximate bayesian inference with expectation propagation

## Installation

Simply type

    pip install choix

The library is under active development, use at your own risk.

## References

- Lucas Maystre and Matthias Grossglauser, [Fast and Accurate Inference of
  Plackett-Luce Models][1], NIPS, 2015
- David R. Hunter. [MM algorithms for generalized Bradley-Terry models][2], The
  Annals of Statistics 32(1):384-406, 2004.
- François Caron and Arnaud Doucet. [Efficient Bayesian Inference for
  Generalized Bradley-Terry models][3]. Journal of Computational and Graphical
  Statistics, 21(1):174-196, 2012.
- Sahand Negahban, Sewoong Oh, and Devavrat Shah, [Iterative Ranking from
  Pair-wise Comparison][4], NIPS 2012
- Hossein Azari Soufiani, William Z. Chen, David C. Parkes, and Lirong Xia,
  [Generalized Method-of-Moments for Rank Aggregation][5], NIPS 2013
- Wei Chu and Zoubin Ghahramani, [Extensions of Gaussian processes for ranking:
  semi-supervised and active learning][6], NIPS 2005 Workshop on Learning to
  Rank.

[1]: https://infoscience.epfl.ch/record/213486/files/fastinference.pdf
[2]: http://sites.stat.psu.edu/~dhunter/papers/bt.pdf
[3]: https://hal.inria.fr/inria-00533638/document
[4]: https://papers.nips.cc/paper/4701-iterative-ranking-from-pair-wise-comparisons.pdf
[5]: https://papers.nips.cc/paper/4997-generalized-method-of-moments-for-rank-aggregation.pdf
[6]: http://www.gatsby.ucl.ac.uk/~chuwei/paper/gprl.pdf
