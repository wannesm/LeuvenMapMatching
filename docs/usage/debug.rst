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


Inspect the matching
--------------------

The best match is available in ``matcher.lattice_best``. This is a list of
``Matching`` objects. For example after running the first example in the introduction:

.. code-block:: python

    >>> matcher.lattice_best
    [Matching<A-B-0-0>,
     Matching<A-B-1-0>,
     Matching<A-B-2-0>,
    ...

A matching object summarizes its information as a tuple with three values if
the best match is with a vertex: <label-observation-nonemitting>. And a tuple
with four values if the best match is with an edge: <labelstart-labelend-observation-nonemitting>.

In the example above, the first observation (with index 0) is matched to a point on the edge
A-B. If you want to inspect the exact locations, you can query the ``Segment``
objects that express the observation and map: ``matching.edge_o`` and ``matching.edge_m``.

.. code-block:: python

   >>> match = matcher.lattice_best[0]
   >>> match.edge_m.l1, match.edge_m.l2  # Edge start/end labels
   ('A', 'B')
   >>> match.edge_m.pi  # Best point on A-B edge
   (1.0, 1.0)
   >>> match.edge_m.p1, match.edge_m.p2  # Locations of A and B
   ((1, 1), (1, 3))
   >>> match.edge_o.l1, match.edge_o.l2  # Observation
   ('O0', None)
   >>> match.edge_o.pi  # Location of observation O0, because no second location
   (0.8, 0.7)
   >>> match.edge_o.p1  # Same as pi because no interpolation
   (0.8, 0.7)
