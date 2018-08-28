Incremental matching
====================

Example: Incremental matching
-------------------------------

If the observations are collected in a streaming setting. The matching can also be invoked incrementally.
The lattice will be built further every time a new subsequence of the path is given.

.. code-block:: python

    from leuvenmapmatching.matcher.distance import DistanceMatcher
    from leuvenmapmatching.map.inmemmap import InMemMap

    map_con = InMemMap("mymap", graph={
        "A": ((1, 1), ["B", "C", "X"]),
        "B": ((1, 3), ["A", "C", "D", "K"]),
        "C": ((2, 2), ["A", "B", "D", "E", "X", "Y"]),
        "D": ((2, 4), ["B", "C", "D", "E", "K", "L"]),
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

    matcher = DistanceMatcher(map_con, max_dist=2, obs_noise=1, min_prob_norm=0.5)
    states, _ = matcher.match_incremental(path[:5])
    states, _ = matcher.match_incremental(path[5:], backtrace_len=-1)
    nodes = matcher.path_pred_onlynodes

    print("States\n------")
    print(states)
    print("Nodes\n------")
    print(nodes)
    print("")
    matcher.print_lattice_stats()


If you do not want to store the entire lattice, you can create a new Matcher object using the
:meth:`copy_lastinterface` before running the incremental matching. This new object will only
contain the last part of the lattice.

.. code-block:: python

    matcher = DistanceMatcher(map_con, max_dist=2, obs_noise=1, min_prob_norm=0.5)
    states, _ = matcher.match_incremental(path[:5])
    matcher = matcher.copy_lastinterface()
    states, _ = matcher.match_incremental(path[5:], backtrace_len=-1)
