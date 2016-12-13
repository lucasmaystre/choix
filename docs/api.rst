API Reference
=============

Functions that :ref:`generate parameters and data <generators>`.

.. autosummary::
   :nosignatures:

   choix.generate_pairwise
   choix.generate_rankings
   choix.footrule_dist

Functions that :ref:`process pairwise comparisons <process-pairwise>`.

.. autosummary::
   :nosignatures:

   choix.lsr_pairwise
   choix.ilsr_pairwise
   choix.opt_pairwise
   choix.ep_pairwise
   choix.mm_pairwise
   choix.log_likelihood_pairwise

Functions that :ref:`process rankings <process-rankings>`.

.. autosummary::
   :nosignatures:

   choix.lsr_rankings
   choix.ilsr_rankings
   choix.log_likelihood_rankings

Functions that :ref:`process top-1 lists <process-top1>`.

.. autosummary::
   :nosignatures:

   choix.lsr_top1
   choix.ilsr_top1
   choix.log_likelihood_top1


.. _generators:

Generators
----------

.. autofunction:: choix.generate_pairwise
.. autofunction:: choix.generate_rankings
.. autofunction:: choix.footrule_dist


.. _process-pairwise:

Processing pairwise comparisons
-------------------------------

.. autofunction:: choix.lsr_pairwise
.. autofunction:: choix.ilsr_pairwise
.. autofunction:: choix.opt_pairwise
.. autofunction:: choix.ep_pairwise
.. autofunction:: choix.mm_pairwise
.. autofunction:: choix.log_likelihood_pairwise


.. _process-rankings:

Processing rankings
-------------------

.. autofunction:: choix.lsr_rankings
.. autofunction:: choix.ilsr_rankings
.. autofunction:: choix.log_likelihood_rankings


.. _process-top1:

Processing top-1 lists
----------------------

.. autofunction:: choix.lsr_top1
.. autofunction:: choix.ilsr_top1
.. autofunction:: choix.log_likelihood_top1
