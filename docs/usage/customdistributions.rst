Custom probability distributions
================================

You can use your own custom probability distributions for the transition and emission probabilities.
This is achieved by inheriting from the :class:`Matcher` class.

Transition probability distribution
-----------------------------------

Overwrite the :meth:`logprob_trans` method.

For example, if you want to use a uniform distribution over the possible road segments:

.. code-block:: python

   def logprob_trans(self, prev_m, next_label=None, next_pos=None):
       return -math.log(len(self.matcher.map.nodes_nbrto(self.edge_m.last_point())))


Emission probability distribution
---------------------------------

Overwrite the :meth:`logprob_obs` method for emitting nodes and the :meth:`logprob_obs_ne` method for
non-emitting nodes. These methods are given the closest distance as `dist`, the previous :class:`Matching` object
in the lattice, the state as `edge_m`, and the observation as `edge_o`. The latter two are :class:`Segment` objects
that can represent either a segment or a point.
Each segment also has a project point which is the point on the segment that is the closest point.

For example, a simple step function with more tolerance for non-emitting nodes:

.. code-block:: python

   def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o):
       if dist < 10:
           return -math.log(10)
       return -np.inf

   def logprob_obs(self, dist, prev_m, new_edge_m, new_edge_o):
       if dist < 50:
           return -math.log(50)
       return -np.inf


Note that an emission probability can be given for a non-emitting node. This allows you to rank non-emitting nodes
even when no observations are available. It will then insert pseudo-observations on the line between the previous
and next observations.
To have a pure non-emitting node, the `logprob_obs_ne` method should always return 0.


Custom lattice objects
----------------------

If you need to store additional information in the lattice, inherit from the :class:`Matching` class and
pass your custom object to the :class:`Matcher` object.

.. code-block:: python

   class MyMatching(Matching):
       ...

   matcher = MyMatcher(mapdb, non_emitting_states=True, only_edges=True, matching=MyMatching)

