Custom probability distributions
================================

You can use your own custom probability distributions for the transition and emission probabilities.
This is achieved by inheriting from the :class:`BaseMatcher` class.

Examples are available in the :class:`SimpleMatching` class and :class:`DistanceMatching` class.
The latter implements a variation based on Newson and Krumm (2009).

Transition probability distribution
-----------------------------------

Overwrite the :meth:`logprob_trans` method.

For example, if you want to use a uniform distribution over the possible road segments:

.. code-block:: python

   def logprob_trans(self, prev_m, edge_m, edge_o, is_prev_ne, is_next_ne):
       return -math.log(len(self.matcher.map.nodes_nbrto(self.edge_m.last_point())))

Note that ``prev_m.edge_m`` and ``edge_m`` are not necessarily connected. For example if the ``Map`` object
returns a neighbor state that is not connected in the roadmap. This functionality is used to allow switching lanes.


Emission probability distribution
---------------------------------

Overwrite the :meth:`logprob_obs` method for non-emitting nodes.
These methods are given the closest distance as `dist`, the previous :class:`Matching` object
in the lattice, the state as `edge_m`, and the observation as `edge_o`. The latter two are :class:`Segment` objects
that can represent either a segment or a point.
Each segment also has a project point which is the point on the segment that is the closest point.

For example, a simple step function with more tolerance for non-emitting nodes:

.. code-block:: python

   def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o, is_ne):
       if is_ne:
           if dist < 50:
               return -math.log(50)
       else:
           if dist < 10:
               return -math.log(10)
       return -np.inf

Note that an emission probability can be given for a non-emitting node. This allows you to rank non-emitting nodes
even when no observations are available. It will then insert pseudo-observations on the line between the previous
and next observations.
To have a pure non-emitting node, the `logprob_obs` method should always return 0 if the
``is_ne`` argument is true.


Custom lattice objects
----------------------

If you need to store additional information in the lattice, inherit from the :class:`Matching` class and
pass your custom object to the :class:`Matcher` object.

.. code-block:: python

   from leuvenmapmatching.map.base import BaseMatching

   class MyMatching(BaseMatching):
       ...

   matcher = MyMatcher(mapdb, matching=MyMatching)

