Types of Data
=============

In order to simplify the code and speed up the implementation of algorithms,
``choix`` assumes that items are identified by consecutive integers ranging
from ``0`` to ``n_items - 1``.

Data processed by the inference algorithms in the library consist of outcomes
of comparisons between subsets of items. Specifically, four types of
observations are supported.


.. _data-pairwise:

Pairwise comparisons
--------------------

In the simplest (and perhaps the most widely-used) case, the data consist of
outcomes of comparisons between *two* items. Mathematically, we represent the
event "item :math:`i` wins over item :math:`j`" as

.. math::

   i \succ j.

In Python, we simply represent this event using a list with two integers:

.. code-block:: python

   [i, j]

By convention, the first element of the list represents the item which wins,
and the second element the item which loses.

The statistical model that ``choix`` postulates for pairwise-comparison
data is usually known as the *Bradley–Terry* model. Given parameters
:math:`\theta_1, \ldots, \theta_n`, and two items :math:`i` and :math:`j`, the
probability of the outcome :math:`i \succ j` is

.. math::

  p(i \succ j) = \frac{e^{\theta_i}}{e^{\theta_i} + e^{\theta_j}}.


.. _data-top1:

Top-1 lists
-----------

Another case arises when the data consist of choices of one item out of a set
containing *several* other items. We call these *top-1 lists*. Compared to
pairwise comparisons, this type of data is no longer restricted to comparing
only two items: comparisons can involve sets of alternatives of any size
between 2 and ``n_items``. We denote the outcome "item :math:`i` is chosen over
items :math:`j, \ldots, k`" as

.. math::

   i \succ \{j, \ldots, k\}.

In Python, we represent this event using a list with two elements:

.. code-block:: python

   [i, {j, ..., k}]

The first element of the list is an integer that represents the "winning" item,
whereas the second element is a set containing the "losing" items. Note that
this set does *not* include the winning item.

The statistical model that ``choix`` uses for these data is a straightforward
extension of the Bradley–Terry model (see, e.g., Luce 1959). Given parameters
:math:`\theta_1, \ldots, \theta_n`, winning item :math:`i` and losing
alternatives :math:`j, k, \ell, \ldots`, the probability of the corresponding
outcome is

.. math::

   p(i \succ \{j, \ldots, k\}) = \frac{e^{\theta_i}}{
       e^{\theta_i} + e^{\theta_j} + \cdots + e^{\theta_k}}.


.. _data-rankings:

Rankings
--------

Instead of observing a single choice, we might have observations that consist
of a *ranking* over a set of alternatives. This leads to a third type of data.
We denote the event "item :math:`i` wins over item :math:`j` ... wins over item
:math:`k`" as

.. math::

   i \succ j \succ \ldots \succ k.

In Python, we represent this as a list:

.. code-block:: python

   [i, j, ..., k]

The list contains the subset of items in decreasing order of preference. For
example, the list ``[2, 0, 4]`` corresponds to a ranking where ``2`` is first,
``0`` is second, and ``4`` is third.

In this case, the statistical model that ``choix`` uses is usually referred to
as the *Plackett–Luce* model. Given parameters :math:`\theta_1, \ldots,
\theta_n` and items :math:`i, j, \ldots, k`, the probability of a given ranking
is

.. math::

   p(i \succ j \succ \ldots \succ k) =
       \frac{e^{\theta_i}}{e^{\theta_i} + e^{\theta_j} + \cdots + e^{\theta_k}}
       \cdot \frac{e^{\theta_j}}{e^{\theta_j} + \cdots + e^{\theta_k}}
       \cdots.

The attentive reader will notice that this probability corresponds to that of
an independent sequence of top-1 lists over the remaining alternatives.


.. _data-network:

Choices in a network
--------------------

The fourth type of data is slightly more involved. It enables the processing of
choices on networks based on marginal observations at the nodes of the network.
The easiest way to get started is to follow  `this notebook
<https://github.com/lucasmaystre/choix/tree/master/notebooks/choicerank-tutorial.ipynb>`__.

We defer to [MG17]_ for a thorough presentation of the observed data and of the
statistical model.
