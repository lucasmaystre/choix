Installing
==========

Most users will probably want to install the latest version hosted on PyPI:

.. code-block:: console

    pip install choix

Developers might want to install the latest version from GitHub:

.. code-block:: console

    git clone https://github.com/lucasmaystre/choix.git
    cd choix
    pip install -e .

The ``-e`` flag makes it possible to edit the code without needing to reinstall
the library afterwards.

Dependencies
------------

``choix`` depends on ``numpy``, ``scipy``, and partially on ``networkx`` (only
for network-related functions). Unit tests depend on ``pytest``.
