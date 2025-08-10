# choix

[![Build package](https://github.com/lucasmaystre/choix/actions/workflows/build.yml/badge.svg)](https://github.com/lucasmaystre/choix/actions/workflows/build.yml)
[![code coverage](https://codecov.io/gh/lucasmaystre/choix/branch/master/graph/badge.svg)](https://codecov.io/gh/lucasmaystre/choix)
[![documentation status](https://readthedocs.org/projects/choix/badge/?version=latest)](http://choix.lum.li/en/latest/?badge=latest)

`choix` is a Python library that provides inference algorithms for models based on Luce\'s choice axiom. These probabilistic models can be used to explain and predict outcomes of comparisons between items.

- **Pairwise comparisons**: when the data consists of comparisons between two items, the model variant is usually referred to as the *Bradley-Terry* model. It is closely related to the Elo rating system used to rank chess players.
- **Partial rankings**: when the data consists of rankings over (a subset of) the items, the model variant is usually referred to as the *Plackett-Luce* model.
- **Top-1 lists**: another variation of the model arises when the data consists of discrete choices, i.e., we observe the selection of one item out of a subset of items.
- **Choices in a network**: when the data consists of counts of the number of visits to each node in a network, the model is known as the *Network Choice Model*.

`choix` makes it easy to infer model parameters from these different types of data, using a variety of algorithms:

- Luce Spectral Ranking
- Minorization-Maximization
- Rank Centrality
- Approximate Bayesian inference with expectation propagation

## Getting started

To install the latest release directly from PyPI, simply type:

    pip install choix

To get started, you might want to explore one of these notebooks:

- [Introduction using pairwise-comparison data](https://github.com/lucasmaystre/choix/blob/master/notebooks/intro-pairwise.ipynb)
- [Case study: analyzing the GIFGIF dataset](https://github.com/lucasmaystre/choix/blob/master/notebooks/gifgif-dataset.ipynb)
- [Using ChoiceRank to understand traffic on a network](https://github.com/lucasmaystre/choix/blob/master/notebooks/choicerank-tutorial.ipynb)
- [Approximate Bayesian inference using EP](https://github.com/lucasmaystre/choix/blob/master/notebooks/ep-example.ipynb)

You can also find more information on the [official documentation](http://choix.lum.li/en/latest/). In particular, the [API reference](http://choix.lum.li/en/latest/api.html) contains a good summary of the library's features.

## References

- Hossein Azari Soufiani, William Z. Chen, David C. Parkes, and Lirong Xia, [Generalized Method-of-Moments for Rank Aggregation](https://papers.nips.cc/paper/4997-generalized-method-of-moments-for-rank-aggregation.pdf), NIPS 2013
- François Caron and Arnaud Doucet. [Efficient Bayesian Inference for Generalized Bradley-Terry models](https://hal.inria.fr/inria-00533638/document). Journal of Computational and Graphical Statistics, 21(1):174-196, 2012.
- Wei Chu and Zoubin Ghahramani, [Extensions of Gaussian processes for ranking: semi-supervised and active learning](http://www.gatsby.ucl.ac.uk/~chuwei/paper/gprl.pdf), NIPS 2005 Workshop on Learning to Rank.
- David R. Hunter. [MM algorithms for generalized Bradley-Terry models](http://sites.stat.psu.edu/~dhunter/papers/bt.pdf), The Annals of Statistics 32(1):384-406, 2004.
- Ravi Kumar, Andrew Tomkins, Sergei Vassilvitskii and Erik Vee,
  [Inverting a Steady-State](http://theory.stanford.edu/~sergei/papers/wsdm15-cset.pdf), WSDM 2015.
- Lucas Maystre and Matthias Grossglauser, [Fast and Accurate Inference of Plackett-Luce Models](https://infoscience.epfl.ch/record/213486/files/fastinference.pdf), NIPS, 2015.
- Lucas Maystre and M. Grossglauser, [ChoiceRank: Identifying Preferences from Node Traffic in Networks](https://infoscience.epfl.ch/record/229164/files/choicerank.pdf), ICML 2017.
- Sahand Negahban, Sewoong Oh, and Devavrat Shah, [Iterative Ranking from Pair-wise Comparison](https://papers.nips.cc/paper/4701-iterative-ranking-from-pair-wise-comparisons.pdf), NIPS 2012.
