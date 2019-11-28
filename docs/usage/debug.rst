Debug
=====

Increasing the verbosity level
------------------------------

To inspect the intermediate steps that the algorithm take, you can increase
the verbosity level of the package. For example:

.. code-block:: python

    import sys
    import logging
    import leuvenmapmatching
    logger = leuvenmapmatching.logger

    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

