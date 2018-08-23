#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_path_onlyedges
~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sys
import os
import logging
from pathlib import Path
import leuvenmapmatching as mm
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.simple import SimpleMatcher


logger = mm.logger
directory = None


def test_path1():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['A', 'B', 'D', 'E', 'F']
    mapdb = InMemMap("map", graph={
        "A": ((1, 1), ["B", "C"]),
        "B": ((1, 3), ["A", "C", "D"]),
        "C": ((2, 2), ["A", "B", "D", "E"]),
        "D": ((2, 4), ["B", "C", "D", "E"]),
        "E": ((3, 3), ["C", "D", "F"]),
        "F": ((3, 5), ["D", "E"])
    }, use_latlon=False)

    matcher = SimpleMatcher(mapdb, max_dist=None, min_prob_norm=None,
                            only_edges=True, non_emitting_states=False)
    matcher.match(path, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        print("Lattice best:")
        for m in matcher.lattice_best:
            print(m)
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_onlyedges_path1.png"))
    assert path_pred == path_sol, f"Paths not equal:\n{path_pred}\n{path_sol}"


def test_path3():
    path = [(3.0, 3.2), (3.1, 3.8), (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['E', 'F']
    mapdb = InMemMap("map", graph={
        "E": ((3, 3), ["F"]),
        "F": ((3, 5), ["E"]),
    }, use_latlon=False)

    matcher = SimpleMatcher(mapdb, max_dist=None, min_prob_norm=0.0001,
                            max_dist_init=1, obs_noise=0.25, obs_noise_ne=10,
                            non_emitting_states=True,
                            only_edges=True)
    matcher.match(path, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        with (directory / 'lattice.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_onlyedges_path3.png"))
        print("Path through lattice:\n" + "\n".join(m.label for m in matcher.lattice_best))
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    test_path1()
    # test_path3()
