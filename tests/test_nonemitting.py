#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_nonemitting
~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2017-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sys
import os
import logging
from pathlib import Path

try:
    import leuvenmapmatching as mm
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    import leuvenmapmatching as mm
from leuvenmapmatching.matcher.distance import DistanceMatcher, DistanceMatching
from leuvenmapmatching.matcher.simple import SimpleMatcher
from leuvenmapmatching.map.inmem import InMemMap

MYPY = False
if MYPY:
    from typing import Tuple


logger = mm.logger
directory = None


def setup_map():
    path1 = [(1.8, 0.1), (1.8, 3.5), (3.0, 4.9)]  # More nodes than observations
    path2 = [(1.8, 0.1), (1.8, 2.0), (1.8, 3.5), (3.0, 4.9)]
    path_sol = ['X', 'C', 'D', 'F']
    mapdb = InMemMap("map", graph={
        "A": ((1, 1), ["B", "C", "X"]),
        "B": ((1, 3), ["A", "C", "D", "K"]),
        "C": ((2, 2), ["A", "B", "D", "E", "X", "Y"]),
        "D": ((2, 4), ["B", "C", "E", "K", "L", "F"]),
        "E": ((3, 3), ["C", "D", "F", "Y"]),
        "F": ((3, 5), ["D", "E", "L"]),
        "X": ((2, 0), ["A", "C", "Y"]),
        "Y": ((3, 1), ["X", "C", "E"]),
        "K": ((1, 5), ["B", "D", "L"]),
        "L": ((2, 6), ["K", "D", "F"])
    }, use_latlon=False)
    return mapdb, path1, path2, path_sol


def visualize_map(pathnb=1):
    mapdb, path1, path2, path_sol = setup_map()
    import leuvenmapmatching.visualization as mm_vis
    if pathnb == 2:
        path = path2
    else:
        path = path1
    mm_vis.plot_map(mapdb, path=path, show_labels=True,
                    filename=(directory / "test_nonemitting_map.png"))


def test_path1():
    mapdb, path1, path2, path_sol = setup_map()

    matcher = SimpleMatcher(mapdb, max_dist_init=1,
                                  min_prob_norm=0.5,
                                  obs_noise=0.5,
                                  non_emitting_states=True, only_edges=False)
    matcher.match(path1, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        matcher.print_lattice_stats()
        matcher.print_lattice()
        with (directory / 'lattice_path1.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       show_graph=True,
                       filename=str(directory / "test_nonemitting_test_path1.png"))
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


def test_path1_inc():
    mapdb, path1, path2, path_sol = setup_map()

    matcher = SimpleMatcher(mapdb, max_dist_init=1,
                            in_prob_norm=0.5, obs_noise=0.5,
                            non_emitting_states=True, only_edges=False,
                            max_lattice_width=1)

    print('## PHASE 1 ##')
    matcher.match(path1, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        matcher.print_lattice_stats()
        matcher.print_lattice()
        with (directory / 'lattice_path1_inc1.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile, precision=2, render=True)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       show_graph=True,
                       filename=str(directory / "test_nonemitting_test_path1_inc1.png"))

    print('## PHASE 2 ##')
    matcher.increase_max_lattice_width(3, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        matcher.print_lattice_stats()
        matcher.print_lattice()
        with (directory / 'lattice_path1_inc2.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile, precision=2, render=True)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       show_graph=True,
                       filename=str(directory / "test_nonemitting_test_path1_inc2.png"))


    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


def test_path1_dist():
    mapdb, path1, path2, path_sol = setup_map()

    matcher = DistanceMatcher(mapdb, max_dist_init=1,
                              min_prob_norm=0.5,
                              obs_noise=0.5,
                              non_emitting_states=True, only_edges=True)
    matcher.match(path1, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        matcher.print_lattice_stats()
        matcher.print_lattice()
        print("LATTICE BEST")
        for m in matcher.lattice_best:
            print(m)
        with (directory / 'lattice_path1.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True, show_graph=True,
                       filename=str(directory / "test_nonemitting_test_path1_dist.png"))
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


def test_path2():
    mapdb, path1, path2, path_sol = setup_map()

    matcher = SimpleMatcher(mapdb, max_dist_init=1, min_prob_norm=0.5, obs_noise=0.5,
                                  non_emitting_states=True, only_edges=False)
    matcher.match(path2, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        matcher.print_lattice_stats()
        matcher.print_lattice()
        with (directory / 'lattice_path2.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_nonemitting_test_path2.png"))
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


def test_path2_dist():
    mapdb, path1, path2, path_sol = setup_map()

    matcher = DistanceMatcher(mapdb, max_dist_init=1, min_prob_norm=0.5,
                              obs_noise=0.5, dist_noise=0.5,
                              non_emitting_states=True)
    matcher.match(path2, unique=True)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        matcher.print_lattice_stats()
        matcher.print_lattice()
        # with (directory / 'lattice_path2.gv').open('w') as ofile:
        #     matcher.lattice_dot(file=ofile)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_nonemitting_test_path2_dist.png"))
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


def test_path2_incremental():
    mapdb, path1, path2, path_sol = setup_map()

    matcher = SimpleMatcher(mapdb, max_dist_init=1, min_prob_norm=0.5, obs_noise=0.5,
                                  non_emitting_states=True, only_edges=False)
    matcher.match_incremental(path2[:2])
    path_pred_1 = matcher.path_pred_onlynodes
    matcher.match_incremental(path2[2:], backtrace_len=len(path2))
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        matcher.print_lattice_stats()
        matcher.print_lattice()
        with (directory / 'lattice_path2.gv').open('w') as ofile:
            matcher.lattice_dot(file=ofile)
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_nonemitting_test_path2.png"))
    assert path_pred_1 == path_sol[:2], "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)
    assert path_pred == path_sol, "Nodes not equal:\n{}\n{}".format(path_pred, path_sol)


def test_path_duplicate():
    from datetime import datetime
    # A path with two identical points
    path = [(0.8, 0.7), (0.9, 0.7), (1.1, 1.0), (1.2, 1.5), (1.2, 1.5), (1.2, 1.6), (1.1, 2.0),
            (1.1, 2.3), (1.3, 2.9), (1.2, 3.1), (1.5, 3.2), (1.8, 3.5), (2.0, 3.7),
            (2.1, 3.3), (2.4, 3.2), (2.6, 3.1), (2.9, 3.1), (3.0, 3.2), (3.1, 3.8),
            (3.0, 4.0), (3.1, 4.3), (3.1, 4.6), (3.0, 4.9)]

    mapdb = InMemMap("map", graph={
        "A": ((1, 1), ["B", "C"]),
        "B": ((1, 3), ["A", "C", "D"]),
        "C": ((2, 2), ["A", "B", "D", "E"]),
        "D": ((2, 4), ["B", "C", "D", "E"]),
        "E": ((3, 3), ["C", "D", "F"]),
        "F": ((3, 5), ["D", "E"])
    }, use_latlon=False)

    matcher = SimpleMatcher(mapdb, max_dist=None, min_prob_norm=None,
                                  non_emitting_states = True, only_edges=False)

    #Matching with and without timestamps signed to the points
    path_pred = matcher.match(path, unique=False)

    path = [(p1, p2, datetime.fromtimestamp(i)) for i, (p1, p2) in enumerate(path)]
    path_pred_time = matcher.match(path, unique=False)

    if directory:
        from leuvenmapmatching import visualization as mmviz
        matcher.print_lattice_stats()
        matcher.print_lattice()
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True,
                       filename=str(directory / "test_nonemitting_test_path_duplicate.png"))

    # The path should be identical regardless of the timestamps
    assert path_pred == path_pred_time, f"Nodes not equal:\n{path_pred}\n{path_pred_time}"


def test_path3_many_obs():
    path = [(1, 0), (3, -0.1), (3.7, 0.6), (4.5, 0.7),
            (5.5, 1.2), (6.5, 0.88), (7.5, 0.65), (8.5, -0.1),
            (9.8, 0.1),(10.1, 1.9)]
    path_sol = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    mapdb = InMemMap("map", graph={
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
    matcher = SimpleMatcher(mapdb, max_dist_init=0.2, obs_noise=1, obs_noise_ne=10,
                                  non_emitting_states=True)
    matcher.match(path)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True, linewidth=10,
                       show_graph=True, show_lattice=True,
                       filename=str(directory / "test_test_path_ne_3_mo.png"))
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


def test_path3_few_obs_en():
    path = [(1, 0), (7.5, 0.65), (10.1, 1.9)]
    path_sol = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    mapdb = InMemMap("map", graph={
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
    matcher = SimpleMatcher(mapdb, max_dist_init=0.2, obs_noise=1, obs_noise_ne=10,
                                  non_emitting_states=True, only_edges=False)
    matcher.match(path)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True, linewidth=10,
                       filename=str(directory / "test_test_path_ne_3_fo.png"))
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


def test_path3_few_obs_e():
    path = [(1, 0), (7.5, 0.65), (10.1, 1.9)]
    path_sol = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    mapdb = InMemMap("map", graph={
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
    matcher = SimpleMatcher(mapdb, max_dist_init=0.2, obs_noise=1, obs_noise_ne=10,
                                  non_emitting_states=True, only_edges=True)
    matcher.match(path)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        matcher.print_lattice_stats()
        matcher.print_lattice()
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True, linewidth=10,
                       filename=str(directory / "test_test_path_e_3_fo.png"))
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"


def test_path3_dist():
    path = [(0, 1), (0.65, 7.5), (1.9, 10.1)]
    path_sol = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    mapdb = InMemMap("map", graph={
        "A": ((0.00, 1), ["B"]),
        "B": ((0.00, 3), ["A", "C"]),
        "C": ((0.70, 3), ["B", "D"]),
        "D": ((1.00, 5), ["C", "E"]),
        "E": ((1.00, 6), ["D", "F"]),
        "F": ((0.70, 7), ["E", "G"]),
        "G": ((0.00, 8), ["F", "H"]),
        "H": ((0.0, 10), ["G", "I"]),
        "I": ((2.0, 10), ["H"])
    }, use_latlon=False)
    matcher = DistanceMatcher(mapdb, max_dist_init=0.2,
                              obs_noise=0.5, obs_noise_ne=2, dist_noise=0.5,
                              non_emitting_states=True)
    states, lastidx = matcher.match(path)
    path_pred = matcher.path_pred_onlynodes
    if directory:
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher, show_labels=True, show_matching=True, linewidth=2,
                       filename=str(directory / "test_path_3_dist.png"))
    assert path_pred == path_sol, f"Nodes not equal:\n{path_pred}\n{path_sol}"

    for obs_idx, m in enumerate(matcher.lattice_best):  # type: Tuple[int, DistanceMatching]
        state = m.shortkey  # tuple indicating edge
        ne_str = "e" if m.is_emitting() else "ne"  # state is emitting or not
        p1_str = "{:>5.2f}-{:<5.2f}".format(*m.edge_m.pi)  # best matching location on graph
        p2_str = "{:>5.2f}-{:<5.2f}".format(*m.edge_o.pi)  # best matching location on track
        print(f"{obs_idx:<2} | {state} | {ne_str:<2} | {p1_str} | {p2_str}")


if __name__ == "__main__":
    # mm.matching.logger.setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # visualize_map(pathnb=1)
    # test_path1()
    test_path1_inc()
    # test_path1_dist()
    # test_path2()
    # test_path2_dist()
    # test_path2_incremental()
    # test_path_duplicate()
    # test_path3_many_obs()
    # test_path3_few_obs_en()
    # test_path3_few_obs_e()
    # test_path3_dist()
