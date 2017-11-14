.. _regularization:

Notes on Regularization
=======================

In some cases, e.g., if the data is sparse, the iterative algorithms underlying
the parameter inference functions might not converge. A pragmatic solution to
this problem is to add a little bit of regularization.

Inference functions in ``choix`` provide a generic regularization argument:
``alpha``. When :math:`\alpha = 0`, regularization is turned off; setting
:math:`\alpha > 0` turns it on. In practice, if regularization is needed, we
recommend starting with small values (e.g., :math:`10^{-4}`) and increasing the
value if necessary.

Below, we briefly how the regularization parameter is used inside the various
parameter inference functions.


.. _regularization-lsr:

Markov-chain based algorithms
-----------------------------

For Markov-chain based algorithms such Luce Spectral Ranking and Rank
Centrality, :math:`\alpha` is used to initialize the transition rates of the
Markov chain.

In the special case of pairwise-comparison data, this can be loosely understood
as placing an independent Beta prior for each pair of items on the respective
comparison outcome probability.


.. _regularization-mm:

Minorization-maximization algorithms
------------------------------------

In the case of Minorization-maximization algorithms, the exponentiated model
parameters :math:`e^{\theta_1}, \ldots, e^{\theta_n}` are endowed each with an
independent Gamma prior distribution, with scale :math:`\alpha + 1`. See Caron
& Doucet (2012) for details.


.. _regularization-other:

Other algorithms
----------------

The scipy-based optimization functions use an :math:`\ell_2`-regularizer on the
parameters :math:`\theta_1, \ldots, \theta_n`. In other words, the parameters
are endowed each with an independent Gaussian prior with variance :math:`1 /
\alpha`.
