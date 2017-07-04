API Reference
=============

Functions that :ref:`generate parameters and data <generators>`.

.. autosummary::
   :nosignatures:

   choix.generate_pairwise
   choix.generate_rankings
   choix.footrule_dist
   choix.compare

Functions that :ref:`process pairwise comparisons <process-pairwise>`.

.. autosummary::
   :nosignatures:

   choix.lsr_pairwise
   choix.ilsr_pairwise
   choix.rank_centrality
   choix.opt_pairwise
   choix.ep_pairwise
   choix.mm_pairwise
   choix.log_likelihood_pairwise

Functions that :ref:`process rankings <process-rankings>`.

.. autosummary::
   :nosignatures:

   choix.lsr_rankings
   choix.ilsr_rankings
   choix.opt_rankings
   choix.mm_rankings
   choix.log_likelihood_rankings

Functions that :ref:`process top-1 lists <process-top1>`.

.. autosummary::
   :nosignatures:

   choix.lsr_top1
   choix.ilsr_top1
   choix.opt_top1
   choix.mm_top1
   choix.log_likelihood_top1

Functions that :ref:`process choices in a network <process-network>`.

.. autosummary::
   :nosignatures:

   choix.choicerank
   choix.log_likelihood_network


.. _generators:

Generators
----------

.. autofunction:: choix.generate_pairwise
.. autofunction:: choix.generate_rankings
.. autofunction:: choix.footrule_dist
.. autofunction:: choix.compare


.. _process-pairwise:

Processing pairwise comparisons
-------------------------------

.. autofunction:: choix.lsr_pairwise
.. autofunction:: choix.ilsr_pairwise
.. autofunction:: choix.rank_centrality
.. autofunction:: choix.opt_pairwise
.. autofunction:: choix.ep_pairwise
.. autofunction:: choix.mm_pairwise
.. autofunction:: choix.log_likelihood_pairwise


.. _process-rankings:

Processing rankings
-------------------

.. autofunction:: choix.lsr_rankings
.. autofunction:: choix.ilsr_rankings
.. autofunction:: choix.opt_rankings
.. autofunction:: choix.mm_rankings
.. autofunction:: choix.log_likelihood_rankings


.. _process-top1:

Processing top-1 lists
----------------------

.. autofunction:: choix.lsr_top1
.. autofunction:: choix.ilsr_top1
.. autofunction:: choix.opt_top1
.. autofunction:: choix.mm_top1
.. autofunction:: choix.log_likelihood_top1


.. _process-network:

Processing choices in a network
-------------------------------

.. autofunction:: choix.choicerank
.. autofunction:: choix.log_likelihood_network
