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


Inspect the best matching
-------------------------

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

Inspect the matching lattice
----------------------------

All paths through the lattice are available in ``matcher.lattice``.
The lattice is a dictionary with a ``LatticeColumn`` object for each observation
(in case the full path of observations is matched).

For each observation, you can inspect the ``Matching`` objects with:

.. code-block:: python

    >>> matcher.lattice
    {0: <leuvenmapmatching.matcher.base.LatticeColumn at 0x12369bf40>,
     1: <leuvenmapmatching.matcher.base.LatticeColumn at 0x123639dc0>,
     2: <leuvenmapmatching.matcher.base.LatticeColumn at 0x123603f40>,
     ...
    >>> matcher.lattice[0].values_all()
    {Matching<A-B-0-0>,
     Matching<A-B-0-1>,
     Matching<A-C-0-0>,
     ...

To start backtracking you can, for example, see which matching object
for the last element has the highest probability (thus the best match):

.. code-block:: python

    >>> m = max(matcher.lattice[len(path)-1].values_all(), key=lambda m: m.logprob)
    >>> m.logprob
    -0.6835815469734807

The previous matching objects can be queried with. These are only those
matches that are connected to this matchin the lattice (in this case
nodes in the street graph with an edge to the current node):

.. code-block:: python

    >>> m.prev  # Best previous match with a connection (multiple if equal probability)
    {Matching<E-F-20-0>}
    >>> m.prev_other  # All previous matches in the lattice with a connection
    {Matching<C-E-20-0>,
     Matching<D-E-20-0>,
     Matching<F-E-20-0>,
     Matching<Y-E-20-0>}
