#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_bugs
~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import os
import sys
import logging
from pathlib import Path
import csv

import leuvenmapmatching as mm
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.map.sqlite import SqliteMap
from leuvenmapmatching.matcher.simple import SimpleMatcher
from leuvenmapmatching.matcher.distance import DistanceMatcher
import leuvenmapmatching.visualization as mm_viz
MYPY = False
if MYPY:
    from typing import List, Tuple


logger = mm.logger
directory = None


def test_bug1():
    dist = 10
    nb_steps = 20

    map_con = InMemMap("map", graph={
        "A":  ((1, dist), ["B"]),
        "B":  ((2, dist), ["A", "C", "CC"]),
        "C":  ((3, 0), ["B", "D"]),
        "D":  ((4 + dist, 0), ["C", "E"]),
        "CC": ((3, 2 * dist), ["B", "DD"]),
        "DD": ((4 + dist, 2 * dist), ["CC", "E"]),
        "E":  ((5 + dist, dist), ["F", "D", "DD"]),
        "F":  ((6 + dist, dist), ["E", ]),

    }, use_latlon=False)

    i = 10
    path = [(1.1,      2*dist*i/nb_steps),
            (2.1,      2*dist*i/nb_steps),
            (5.1+dist, 2*dist*i/nb_steps),
            (6.1+dist, 2*dist*i/nb_steps)
            # (1, len*i/nb_steps),
            # (2, len*i/nb_steps),
            # (3, len*i/nb_steps)
            ]

    matcher = SimpleMatcher(map_con, max_dist=dist + 1, obs_noise=dist + 1, min_prob_norm=None,
                                  non_emitting_states=True)

    nodes = matcher.match(path, unique=False)
    print("Solution: ", nodes)
    if directory:
        import leuvenmapmatching.visualization as mm_vis
        matcher.print_lattice()
        matcher.print_lattice_stats()
        mm_vis.plot_map(map_con, path=path, nodes=nodes, counts=matcher.node_counts(),
                        show_labels=True, filename=str(directory / "test_bugs_1.png"))


def test_bug2():
    this_path = Path(os.path.realpath(__file__)).parent / "rsrc" / "bug2"
    edges_fn = this_path / "edgesrl.csv"
    nodes_fn = this_path / "nodesrl.csv"
    path_fn = this_path / "path.csv"

    logger.debug(f"Reading map ...")
    mmap = SqliteMap("road_network", use_latlon=True, dir=this_path)

    path = []
    with path_fn.open("r") as path_f:
        reader = csv.reader(path_f, delimiter=',')
        for row in reader:
            lat, lon = [float(coord) for coord in row]
            path.append((lat, lon))
    node_cnt = 0
    with nodes_fn.open("r") as nodes_f:
        reader = csv.reader(nodes_f, delimiter=',')
        for row in reader:
            nid, lonlat, _ = row
            nid = int(nid)
            lon, lat = [float(coord) for coord in lonlat[1:-1].split(",")]
            mmap.add_node(nid, (lat, lon), ignore_doubles=True, no_index=True, no_commit=True)
            node_cnt += 1
    edge_cnt = 0
    with edges_fn.open("r") as edges_f:
        reader = csv.reader(edges_f, delimiter=',')
        for row in reader:
            _eid, nid1, nid2, pid = [int(val) for val in row]
            mmap.add_edge(nid1, nid2, edge_type=0, path=pid, no_index=True, no_commit=True)
            edge_cnt += 1
    logger.debug(f"... done: {node_cnt} nodes and {edge_cnt} edges")
    logger.debug("Indexing ...")
    mmap.reindex_nodes()
    mmap.reindex_edges()
    logger.debug("... done")

    matcher = DistanceMatcher(mmap, min_prob_norm=0.001,
                              max_dist=200, obs_noise=4.07,
                              non_emitting_states=True)
    # path = path[:2]
    nodes, idx = matcher.match(path, unique=True)
    path_pred = matcher.path_pred
    if directory:
        import matplotlib.pyplot as plt
        matcher.print_lattice_stats()
        logger.debug("Plotting post map ...")
        fig = plt.figure(figsize=(100, 100))
        ax = fig.get_axes()
        mm_viz.plot_map(mmap, matcher=matcher, use_osm=True, ax=ax,
                        show_lattice=False, show_labels=True, show_graph=False, zoom_path=True,
                        show_matching=True)
        plt.savefig(str(directory / "test_bug1.png"))
        plt.close(fig)
        logger.debug("... done")


if __name__ == "__main__":
    mm.logger.setLevel(logging.DEBUG)
    mm.logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_bug1()
    test_bug2()
