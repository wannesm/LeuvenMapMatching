Examples
========

Example 1: Simple
-----------------

A first, simple example. Some parameters are given to tune the algorithm.
The ``max_dist`` and ``obs_noise`` are distances that indicate the maximal distance between observation and road
segment and the expected noise in the measurements, respectively.
The ``min_prob_norm`` prunes the lattice in that it drops paths that drop below 0.5 normalized probability.
The probability is normalized to allow for easier reasoning about the probability of a path.
It is computed as the exponential smoothed log probability components instead of the sum as would be the case
for log likelihood.
Because the number of possible paths quickly grows, it is recommended to set the
``max_lattice_width`` argument (available from version 1.0 onwards). It will only continue the search with this number of
possible paths at every step. If no solution is found, this value can be incremented
using the ``increase_max_lattice_width`` method.

.. code-block:: python

    from leuvenmapmatching.matcher.distance import DistanceMatcher
    from leuvenmapmatching.map.inmem import InMemMap

    map_con = InMemMap("mymap", graph={
        "A": ((1, 1), ["B", "C", "X"]),
        "B": ((1, 3), ["A", "C", "D", "K"]),
        "C": ((2, 2), ["A", "B", "D", "E", "X", "Y"]),
        "D": ((2, 4), ["B", "C", "F", "E", "K", "L"]),
        "E": ((3, 3), ["C", "D", "F", "Y"]),
        "F": ((3, 5), ["D", "E", "L"]),
        "X": ((2, 0), ["A", "C", "Y"]),
        "Y": ((3, 1), ["X", "C", "E"]),
        "K": ((1, 5), ["B", "D", "L"]),
        "L": ((2, 6), ["K", "D", "F"])
    }, use_latlon=False)

    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.3, 3.5), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2),
            (3.1, 3.8), (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]

    matcher = DistanceMatcher(map_con, max_dist=2, obs_noise=1, min_prob_norm=0.5, max_lattice_width=5)
    states, _ = matcher.match(path)
    nodes = matcher.path_pred_onlynodes

    print("States\n------")
    print(states)
    print("Nodes\n------")
    print(nodes)
    print("")
    matcher.print_lattice_stats()


Example 2: Non-emitting states
------------------------------

In case there are less observations that states (an assumption of HMMs), non-emittings states allow you
to deal with this. States will be inserted that are not associated with any of the given observations if
this improves the probability of the path.

It is possible to also associate a distribtion over the distance between observations and the non-emitting
states (`obs_noise_ne`). This allows the algorithm to prefer nearby road segments. This value should be
larger than `obs_noise` as it is mapped to the line between the previous and next observation, which does
not necessarily run over the relevant segment. Setting this to infinity is the same as using pure
non-emitting states that ignore observations completely.

.. code-block:: python

    from leuvenmapmatching.matcher.distance import DistanceMatcher
    from leuvenmapmatching.map.inmem import InMemMap
    from leuvenmapmatching import visualization as mmviz

    path = [(1, 0), (7.5, 0.65), (10.1, 1.9)]
    mapdb = InMemMap("mymap", graph={
        "A": ((1, 0.00), ["B"]),
        "B": ((3, 0.00), ["A", "C"]),
        "C": ((4, 0.70), ["B", "D"]),
        "D": ((5, 1.00), ["C", "E"]),
        "E": ((6, 1.00), ["D", "F"]),
        "F": ((7, 0.70), ["E", "G"]),
        "G": ((8, 0.00), ["F", "H"]),
        "H": ((10, 0.0), ["G", "I"]),
        "I": ((10, 2.0), ["H"])
    }, use_latlon=False)
    matcher = DistanceMatcher(mapdb, max_dist_init=0.2, obs_noise=1, obs_noise_ne=10,
                              non_emitting_states=True, only_edges=True)
    states, _ = matcher.match(path)
    nodes = matcher.path_pred_onlynodes

    print("States\n------")
    print(states)
    print("Nodes\n------")
    print(nodes)
    print("")
    matcher.print_lattice_stats()

    mmviz.plot_map(mapdb, matcher=matcher,
                  show_labels=True, show_matching=True
                  filename="output.png"))
