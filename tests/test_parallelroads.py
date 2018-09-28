#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_parallelroads
~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sys
import os
import logging
from pathlib import Path
import leuvenmapmatching as mm
from leuvenmapmatching.map.sqlite import SqliteMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.util.dist_euclidean import lines_parallel


logger = mm.logger
directory = None


def create_map1():
    db = SqliteMap("map", use_latlon=False, dir=directory)
    logger.debug(f"Initialized db: {db}")
    db.add_nodes([
        (1, (1, 1)),
        (2, (1, 2.9)),
        (22, (1, 3.0)),
        (3, (2, 2)),
        (33, (2, 2.1)),
        (4, (2, 4)),
        (5, (3, 3)),
        (6, (3, 5))
    ])
    db.add_edges([
        (1, 2), (1, 3),
        (2, 22), (2, 1),
        (22, 2), (22, 33), (22, 4),
        (3, 33), (3, 1), (3, 2), (3, 5),
        (33, 3),
        (4, 22), (4, 33), (4, 5), (4, 6),
        (5, 3), (5, 4), (5, 6),
        (6, 4), (6, 5)
    ])
    logger.debug(f"Filled db: {db}")
    return db


def create_path1():
    return [(0.9, 2.5), (1.1, 2.75), (1.25, 2.6), (1.4, 2.5), (1.5, 2.4), (1.6, 2.5), (1.4, 2.7), (1.2, 2.9), (1.1, 3.0), (1.3, 3.2)]


def test_parallel():
    result = lines_parallel((1, 2.9), (2, 2), (1, 3.0), (2, 2.1), d=0.1)
    assert result is True


def test_bb1():
    mapdb = create_map1()

    if directory:
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb,
                       show_labels=True, show_graph=True,
                       filename=str(directory / "test_bb.png"))

    mapdb.connect_parallelroads()

    assert mapdb.size() == 8
    coord = mapdb.node_coordinates(2)
    assert coord == (1, 2.9)

    nodes = mapdb.all_nodes(bb=[0.5, 2.5, 1.5, 3.5])
    node_ids = set([nid for nid, _ in nodes])
    assert node_ids == {2, 22}

    edges = mapdb.all_edges(bb=[0.5, 2.5, 1.5, 3.5])
    edge_tuples = set((nid1, nid2) for nid1, _, nid2, _ in edges)
    assert edge_tuples == {(1, 2), (2, 1), (3, 2), (2, 22), (22, 2), (22, 33), (22, 4), (4, 22)}

    nodes = mapdb.nodes_nbrto(2)
    node_ids = set([nid for nid, _ in nodes])
    assert node_ids == {1, 22}

    edges = mapdb.edges_nbrto((1, 2))
    edge_tuples = set((nid1, nid2) for nid1, _, nid2, _ in edges)
    assert edge_tuples == {(2, 22), (2, 1)}

    edges = mapdb.edges_nbrto((3, 2))
    edge_tuples = set((nid1, nid2) for nid1, _, nid2, _ in edges)
    assert edge_tuples == {(22, 33), (2, 22), (2, 1)}


def test_merge1():
    mapdb = create_map1()
    mapdb.connect_parallelroads()

    if directory:
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb,
                       show_labels=True, show_graph=True,
                       filename=str(directory / "test_parallel_merge.png"))


def test_path1():
    mapdb = create_map1()
    mapdb.connect_parallelroads()
    path = create_path1()
    states_sol = [(1, 2), (2, 22), (22, 33), (22, 33), (22, 33), (3, 2), (3, 2), (3, 2), (2, 22), (22, 4)]

    matcher = DistanceMatcher(mapdb, max_dist_init=0.2,
                              obs_noise=0.5, obs_noise_ne=2, dist_noise=0.5,
                              non_emitting_states=True)
    states, _ = matcher.match(path)

    if directory:
        from leuvenmapmatching import visualization as mmviz
        mmviz.plot_map(mapdb, matcher=matcher,
                       show_labels=True, show_graph=True, show_matching=True,
                       filename=str(directory / "test_parallel_merge.png"))
    assert states == states_sol, f"Unexpected states: {states}"


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_parallel()
    # test_merge1()
    # test_path1()
    test_bb1()
