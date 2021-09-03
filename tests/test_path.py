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

sys.path.append("..")
import leuvenmapmatching as mm
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.simple import SimpleMatcher
from leuvenmapmatching.matcher.distance import DistanceMatcher


logger = mm.logger
directory = None


def test_path1():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    # path_sol = ['A', ('A', 'B'), 'B', ('B', 'D'), 'D', ('D', 'E'), 'E', ('E', 'F')]
    path_sol_nodes = ['A', 'B', 'D', 'E', 'F']
    mapdb = InMemMap("map", graph={
        "A": ((1, 1), ["B", "C"]),
        "B": ((1, 3), ["A", "C", "D"]),
        "C": ((2, 2), ["A", "B", "D", "E"]),
        "D": ((2, 4), ["B", "C", "D", "E"]),
        "E": ((3, 3), ["C", "D", "F"]),
        "F": ((3, 5), ["D", "E"])
    }, use_latlon=False)

    matcher = SimpleMatcher(mapdb, max_dist=None, min_prob_norm=None,
                            non_emitting_states=False, only_edges=False)
    path_pred, _ = matcher.match(path, unique=True)
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       show_graph=True, show_lattice=True,
                       filename=str(directory / "test_path1.png"))
    # assert path_pred == path_sol, f"Paths not equal:\n{path_pred}\n{path_sol}"
    nodes_pred = matcher.path_pred_onlynodes
    assert nodes_pred == path_sol_nodes, f"Nodes not equal:\n{nodes_pred}\n{path_sol_nodes}"


def test_path1_dist():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    # path_sol = ['A', ('A', 'B'), 'B', ('B', 'D'), 'D', ('D', 'E'), 'E', ('E', 'F')]
    path_sol_nodes = ['A', 'B', 'D', 'E', 'F']
    mapdb = InMemMap("map", graph={
        "A": ((1, 1), ["B", "C"]),
        "B": ((1, 3), ["A", "C", "D"]),
        "C": ((2, 2), ["A", "B", "D", "E"]),
        "D": ((2, 4), ["B", "C", "D", "E"]),
        "E": ((3, 3), ["C", "D", "F"]),
        "F": ((3, 5), ["D", "E"])
    }, use_latlon=False)

    matcher = DistanceMatcher(mapdb, max_dist=None, min_prob_norm=None,
                              obs_noise=0.5,
                              non_emitting_states=False)
    matcher.match(path)
    if directory:
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True, show_graph=True,
                       filename=str(directory / "test_path1_dist.png"))
    nodes_pred = matcher.path_pred_onlynodes
    assert nodes_pred == path_sol_nodes, f"Nodes not equal:\n{nodes_pred}\n{path_sol_nodes}"


def test_path2():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    # path_sol = ['A', ('A', 'B'), 'B', ('B', 'D'), 'D', ('D', 'E'), 'E', ('E', 'F')]
    path_sol_nodes = ['A', 'B', 'D', 'E', 'F']
    mapdb = InMemMap("map", graph={
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

    matcher = SimpleMatcher(mapdb, max_dist=None, min_prob_norm=0.001,
                            non_emitting_states=False, only_edges=False,
                            max_lattice_width=3)
    path_pred, _ = matcher.match(path, unique=True)
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       show_lattice=True, show_graph=True,
                       filename=str(directory / "test_path2.png"))
    # assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)
    nodes_pred = matcher.path_pred_onlynodes
    assert nodes_pred == path_sol_nodes, f"Nodes not equal:\n{nodes_pred}\n{path_sol_nodes}"


def test_path2_inc():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    # path_sol = ['A', ('A', 'B'), 'B', ('B', 'D'), 'D', ('D', 'E'), 'E', ('E', 'F')]
    path_sol_nodes = ['A', 'B', 'D', 'E', 'F']
    mapdb = InMemMap("map", graph={
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

    ## Phase 1
    print('=== PHASE 1 ===')
    matcher = SimpleMatcher(mapdb, max_dist=None, min_prob_norm=0.001,
                            non_emitting_states=False, only_edges=False,
                            max_lattice_width=1)
    path_pred, _ = matcher.match(path, unique=True)
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        with (directory / 'test_path2_inc_1.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile, precision=2, render=True)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       show_lattice=True, show_graph=True,
                       filename=str(directory / "test_path2_inc_1.png"))

    ## Next phases
    for phase_nb, phase_width in enumerate([2, 3]):
        print(f'=== PHASE {phase_nb + 2} ===')
        path_pred, _ = matcher.increase_max_lattice_width(phase_width, unique=True)
        if directory:
            matcher.print_lattice_stats()
            matcher.print_lattice()
            from leuvenmapmatching import visualization as mmviz
            with (directory / f'test_path2_inc_{phase_nb + 2}.gv').open('w') as ofile:
                matcher.lattice_dot(file=ofile, precision=2, render=True)
            mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                           show_lattice=True, show_graph=True,
                           filename=str(directory / f"test_path2_inc_{phase_nb + 2}.png"))

    # assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)
    nodes_pred = matcher.path_pred_onlynodes
    assert nodes_pred == path_sol_nodes, f"Nodes not equal:\n{nodes_pred}\n{path_sol_nodes}"


def test_path2_dist():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol_nodes = ['X', 'A', 'B', 'D', 'E', 'F']
    mapdb = InMemMap("map", graph={
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

    matcher = DistanceMatcher(mapdb, max_dist=None, min_prob_norm=0.001,
                              obs_noise=0.5,
                              non_emitting_states=False)
    matcher.match(path, unique=True)
    if directory:
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher,
                       show_labels=True, show_matching=True, show_graph=True,
                       filename=str(directory / "test_path2_dist.png"))
    nodes_pred = matcher.path_pred_onlynodes
    assert nodes_pred == path_sol_nodes, f"Nodes not equal:\n{nodes_pred}\n{path_sol_nodes}"


def test_path_outlier():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['A', 'B', 'D', 'C', 'D', 'E', 'F']
    path.insert(13, (2.3, 1.8))
    mapdb = InMemMap("map", graph={
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

    matcher = SimpleMatcher(mapdb, max_dist=None, min_prob_norm=0.0001,
                            max_dist_init=1, obs_noise=0.5, obs_noise_ne=10,
                            non_emitting_states=True)
    _, last_idx = matcher.match(path, unique=True)
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
    assert last_idx == len(path) - 1
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


def test_path_outlier2():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path.insert(13, (2.3, -3.0))
    mapdb = InMemMap("map", graph={
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

    matcher = DistanceMatcher(mapdb, max_dist=None, min_prob_norm=0.1,
                            max_dist_init=1, obs_noise=0.25, obs_noise_ne=1,
                            non_emitting_states=True)
    _, last_idx = matcher.match(path, unique=True)
    if directory:
        # matcher.print_lattice_stats()
        # matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        # with (directory / 'lattice.gv').open('w') as ofile:
        #     matcher.lattice_dot(file=ofile)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_path_outlier2.png"))
    assert last_idx == 12


def test_path_outlier_dist():
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['A', 'B', 'D', 'C', 'E', 'F']
    path.insert(13, (2.3, 1.8))
    mapdb = InMemMap("map", graph={
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

    matcher = DistanceMatcher(mapdb, max_dist=None, min_prob_norm=0.0001,
                              max_dist_init=1, obs_noise=0.5, obs_noise_ne=10,
                              non_emitting_states=True)
    matcher.match(path)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher,
                       show_labels=True, show_matching=True, show_graph=True,
                       filename=str(directory / "test_path_outlier_dist.png"))
    # TODO: Smoothing the observation distances could eliminate the outlier
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


def test_path3():
    path = [(3.0, 3.2), (3.1, 3.8), (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['E', 'F']
    mapdb = InMemMap("map", graph={
        "E": ((3, 3), ["F"]),
        "F": ((3, 5), ["E"]),
    }, use_latlon=False)

    matcher = SimpleMatcher(mapdb, max_dist=None, min_prob_norm=0.0001,
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


def test_path3_dist():
    path = [(3.0, 3.2), (3.1, 3.8), (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]
    path_sol = ['E', 'F']
    mapdb = InMemMap("map", graph={
        "E": ((3, 3), ["F"]),
        "F": ((3, 5), ["E"]),
    }, use_latlon=False)

    matcher = DistanceMatcher(mapdb, max_dist=None, min_prob_norm=0.0001,
                              max_dist_init=1, obs_noise=0.25, obs_noise_ne=10,
                              non_emitting_states=True)
    matcher.match(path, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_path3_dist.png"))
        print("Path through lattice:\n" + "\n".join(m.label for m in matcher.lattice_best))
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_path1()
    # test_path1_dist()
    # test_path2()
    # test_path2_inc()
    test_path2_dist()
    # test_path_outlier()
    # test_path_outlier2()
    # test_path_outlier_dist()
    # test_path3()
    # test_path3_dist()
