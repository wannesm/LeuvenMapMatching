#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_path
~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2017-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sys
import os
import logging
from pathlib import Path
import leuvenmapmatching as mm


directory = None


def test_path1():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['A', ('A', 'B'), 'B', ('B', 'D'), 'D', ('D', 'E'), 'E', ('E', 'F')]
    path_sol_nodes = ['A', 'B', 'D', 'E', 'F']
    mapdb = mm.map.InMemMap(graph=[
        ("A", (1, 1), ["B", "C"]),
        ("B", (1, 3), ["A", "C", "D"]),
        ("C", (2, 2), ["A", "B", "D", "E"]),
        ("D", (2, 4), ["B", "C", "D", "E"]),
        ("E", (3, 3), ["C", "D", "F"]),
        ("F", (3, 5), ["D", "E"])
    ], use_latlon=False)

    matcher = mm.matching.Matcher(mapdb, max_dist=None, min_prob_norm=None)
    path_pred, _ = matcher.match(path, unique=True)
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_path1.png"))
    assert path_pred == path_sol, f"Paths not equal:\n{path_pred}\n{path_sol}"
    nodes_pred = matcher.path_pred_onlynodes
    assert nodes_pred == path_sol_nodes, f"Nodes not equal:\n{nodes_pred}\n{path_sol_nodes}"


def test_path2():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['A', ('A', 'B'), 'B', ('B', 'D'), 'D', ('D', 'E'), 'E', ('E', 'F')]
    mapdb = mm.map.InMemMap(graph=[
        ("A", (1, 1), ["B", "C", "X"]),
        ("B", (1, 3), ["A", "C", "D", "K"]),
        ("C", (2, 2), ["A", "B", "D", "E", "X", "Y"]),
        ("D", (2, 4), ["B", "C", "D", "E", "K", "L"]),
        ("E", (3, 3), ["C", "D", "F", "Y"]),
        ("F", (3, 5), ["D", "E", "L"]),
        ("X", (2, 0), ["A", "C", "Y"]),
        ("Y", (3, 1), ["X", "C", "E"]),
        ("K", (1, 5), ["B", "D", "L"]),
        ("L", (2, 6), ["K", "D", "F"])
    ], use_latlon=False)

    matcher = mm.matching.Matcher(mapdb, max_dist=None, min_prob_norm=0.001)
    path_pred, _ = matcher.match(path, unique=True)
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_path2.png"))
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


def test_path_outlier():
    # TODO: Should we deal better with the outlier by using a smarter matcher?
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['A', 'B', 'D', 'C', 'E', 'F']
    path.insert(13, (2.3, 1.8))
    mapdb = mm.map.InMemMap(graph=[
        ("A", (1, 1), ["B", "C", "X"]),
        ("B", (1, 3), ["A", "C", "D", "K"]),
        ("C", (2, 2), ["A", "B", "D", "E", "X", "Y"]),
        ("D", (2, 4), ["B", "C", "D", "E", "K", "L"]),
        ("E", (3, 3), ["C", "D", "F", "Y"]),
        ("F", (3, 5), ["D", "E", "L"]),
        ("X", (2, 0), ["A", "C", "Y"]),
        ("Y", (3, 1), ["X", "C", "E"]),
        ("K", (1, 5), ["B", "D", "L"]),
        ("L", (2, 6), ["K", "D", "F"])
    ], use_latlon=False)

    matcher = mm.matching.Matcher(mapdb, max_dist=None, min_prob_norm=0.0001,
                                  max_dist_init=1, obs_noise=0.5, obs_noise_ne=10,
                                  non_emitting_states=True)
    matcher.match(path, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        with (directory / 'lattice.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_path_outlier.png"))
        print("Path through lattice:\n" + "\n".join(m.label for m in matcher.lattice_best))
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


def test_path3():
    path = [(3.0, 3.2), (3.1, 3.8), (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['E', 'F']
    mapdb = mm.map.InMemMap(graph=[
        ("E", (3, 3), ["F"]),
        ("F", (3, 5), ["E"]),
    ], use_latlon=False)

    matcher = mm.matching.Matcher(mapdb, max_dist=None, min_prob_norm=0.0001,
                                  max_dist_init=1, obs_noise=0.25, obs_noise_ne=10,
                                  non_emitting_states=True)
    matcher.match(path, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        with (directory / 'lattice.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_path3.png"))
        print("Path through lattice:\n" + "\n".join(m.label for m in matcher.lattice_best))
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


if __name__ == "__main__":
    mm.matching.logger.setLevel(logging.DEBUG)
    mm.matching.logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    test_path1()
    # test_path2()
    # test_path_outlier()
    # test_path3()
