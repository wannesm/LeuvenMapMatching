#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_path_newsonkrumm2009
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the data available at:
https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/

Notes:

* There is a 'bug' in the map available from the website.
  Multiple segments (streets) in the map are not connected but have overlappen, but
  disconnected, nodes.
  For example, the following nodes are on the same location and
  should be connected because the given path runs over this road:
  - 884147801204 and 884148400033
  - 884148100260 and 884148001002
* The path is missing a number of observations. For those parts non-emitting nodes are required.
  This occurs at:
  - 2770:2800 (index 2659 is start)
  - 2910:2929

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import os
import sys
import logging
import pickle
from pathlib import Path
import csv
from datetime import datetime
import pytest
import leuvenmapmatching as mm
from leuvenmapmatching.matcher import base
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
import leuvenmapmatching.visualization as mm_viz
MYPY = False
if MYPY:
    from typing import List, Tuple


logger = mm.logger
this_path = Path(os.path.realpath(__file__)).parent / "rsrc" / "newson_krumm_2009"
gps_data = this_path / "gps_data.txt"
gps_data_pkl = gps_data.with_suffix(".pkl")
gps_data_xy_pkl = this_path / "gps_data_xy.pkl"
ground_truth_route = this_path / "ground_truth_route.txt"
road_network = this_path / "road_network.txt"
road_network_pkl = road_network.with_suffix(".pkl")
road_network_xy_pkl = this_path / "road_network_xy.pkl"

directory = None
base.default_label_width = 34


def read_gps(route_fn):
    route = []
    with route_fn.open("r") as route_f:
        reader = csv.reader(route_f, delimiter='\t')
        next(reader)
        for row in reader:
            date, time, lat, lon = row[:4]
            date_str = date + " " + time
            ts = datetime.strptime(date_str, '%d-%b-%Y %H:%M:%S')
            lat = float(lat)
            lon = float(lon)
            route.append((lat, lon, ts))
    logger.debug(f"Read GPS trace of {len(route)} points")
    return route


def read_nodes(nodes_fn):
    nodes = []
    with nodes_fn.open("r") as nodes_f:
        reader = csv.reader(nodes_f, delimiter='\t')
        next(reader)
        for row in reader:
            nodeid, trav = row[:2]
            nodeid = int(nodeid)
            trav = int(trav)
            nodes.append((nodeid, trav))
    logger.debug(f"Read correct trace of {len(nodes)} nodes")
    return nodes


def parse_linestring(line):
    # type: (str) -> List[Tuple[float, float]]
    line = line[line.index("(") + 1:line.index(")")]
    latlons = []
    for lonlat in line.split(", "):
        lon, lat = lonlat.split(" ")
        latlons.append((float(lat), float(lon)))
    return latlons


def read_map(map_fn):
    mmap = InMemMap("road_network", use_latlon=True, use_rtree=False,
                    index_edges=True, dir=this_path)
    node_cnt = 0
    edge_cnt = 0
    # new_node_id = 1000000000000
    new_node_id = 1
    with map_fn.open("r") as map_f:
        reader = csv.reader(map_f, delimiter='\t')
        next(reader)
        for row in reader:
            eid, nf, nt, twoway, speed, length,  innernodes = row
            nf = int(nf)
            nt = int(nt)
            length = int(length)
            innernodes = parse_linestring(innernodes)
            # Add nodes to map
            mmap.add_node(nf, innernodes[0])
            mmap.add_node(nt, innernodes[-1])
            node_cnt += 2
            prev_node = nf
            assert(length < 1000)
            for idx, innernode in enumerate(innernodes[1:-1]):
                # innernode_id = nf * 1000 + idx
                innernode_id = new_node_id
                new_node_id += 1
                mmap.add_node(innernode_id, innernode)
                node_cnt += 1
                mmap.add_edge(prev_node, innernode_id)
                mmap.add_edge(innernode_id, prev_node)
                edge_cnt += 2
                prev_node = innernode_id
            mmap.add_edge(prev_node, nt)
            mmap.add_edge(nt, prev_node)
    logger.debug(f"Read map with {node_cnt} nodes and {edge_cnt} edges")
    assert(new_node_id < 100000000000)
    return mmap


def correct_map(mmap):
    """Add edges between nodes with degree > 2 that are on the exact same location."""
    def correct_edge(label, others):
        if label < 100000000000:
            return
        logger.info(f"Add connections between {label} and {others}")
        for other in others:
            if other > 100000000000:
                mmap.add_edge(label, other)
    mmap.find_duplicates(func=correct_edge)


def load_data():
    max_route_length = None  #200

    # Nodes
    nodes = read_nodes(ground_truth_route)

    # Map
    if road_network_pkl.exists() and road_network_xy_pkl.exists():
        map_con_latlon = InMemMap.from_pickle(road_network_pkl)
        logger.debug(f"Read latlon road network from file ({map_con_latlon.size()} nodes)")
        map_con = InMemMap.from_pickle(road_network_xy_pkl)
        logger.debug(f"Read xy road network from file ({map_con.size()} nodes)")
    else:
        map_con_latlon = read_map(road_network)
        map_con_latlon.dump()
        logger.debug(f"Saved latlon road network to file ({map_con_latlon.size()} nodes)")
        map_con = map_con_latlon.to_xy(name="road_network_xy", use_rtree=True)
        correct_map(map_con)
        map_con.dump()
        logger.debug(f"Saved xy road network to file ({map_con.size()} nodes)")

    # Route
    if gps_data_pkl.exists() and gps_data_xy_pkl.exists():
        with gps_data_pkl.open("rb") as ifile:
            route_latlon = pickle.load(ifile)
        with gps_data_xy_pkl.open("rb") as ifile:
            route = pickle.load(ifile)
        logger.debug(f"Read gps route from file ({len(route)} points)")
    else:
        route_latlon = read_gps(gps_data)
        if max_route_length:
            route_latlon = route_latlon[:max_route_length]
        with gps_data_pkl.open("wb") as ofile:
            pickle.dump(route_latlon, ofile)
        route = [map_con.latlon2yx(lat, lon) for lat, lon, _ in route_latlon]
        with gps_data_xy_pkl.open("wb") as ofile:
            pickle.dump(route, ofile)

    return nodes, map_con, map_con_latlon, route, route_latlon


def test_route_slice1():
    if directory:
        import matplotlib.pyplot as plt
    nodes, map_con, map_con_latlon, route, route_latlon = load_data()
    zoom_path = True

    matcher = DistanceMatcher(map_con, min_prob_norm=0.001,
                              max_dist=200,
                              dist_noise=6, dist_noise_ne=12,
                              obs_noise=30, obs_noise_ne=150,
                              non_emitting_states=True)
    route_slice = route[2657:2662]
    matcher.match(route_slice)
    path_pred = matcher.path_pred_onlynodes
    path_sol = [172815, 172816, 172817, 172818, 172819, 172820, 172821, 172822, 172823, 172824,
                172825, 172826, 172827, 172828, 172829, 172830, 884148100261, 172835, 172836,
                172837, 884148100254, 172806, 884148100255, 172807]  # Can change when building db
    assert len(path_pred) == len(path_sol)


def test_bug1():
    map_con_ll = InMemMap("map", graph={
        "A": ((47.590439915657, -122.238368690014), ["B"]),
        "B": ((47.5910192728043, -122.239519357681), ["C"]),
        "C": ((47.5913706421852, -122.240168452263), [])
    }, use_latlon=True)
    map_con = map_con_ll.to_xy(name="road_network_xy")
    path_ll = [
        # (47.59043333, -122.2384167),
        (47.59058333, -122.2387),
        (47.59071667, -122.2389833),
        (47.59086667, -122.2392667),
        (47.59101667, -122.23955),
        (47.59115,    -122.2398333)
    ]
    path = [map_con.latlon2yx(lat, lon) for lat, lon in path_ll]
    path_sol = [('A', 'B'), ('B', 'C')]
    matcher = DistanceMatcher(map_con, min_prob_norm=0.001,
                              max_dist=200, obs_noise=4.07,
                              non_emitting_states=True)
    matcher.match(path, unique=True)
    path_pred = matcher.path_pred
    if directory:
        import matplotlib.pyplot as plt
        matcher.print_lattice_stats()
        logger.debug("Plotting post map ...")
        fig = plt.figure(figsize=(100,100))
        ax = fig.get_axes()
        mm_viz.plot_map(map_con, matcher=matcher, use_osm=True, ax=ax,
                        show_lattice=False, show_labels=True, show_graph=True, zoom_path=True,
                        show_matching=True,
                        coord_trans=map_con.yx2latlon)
        plt.savefig(str(directory / "test_newson_bug1.png"))
        plt.close(fig)
        logger.debug("... done")
    assert path_pred == path_sol, f"Edges not equal:\n{path_pred}\n{path_sol}"


@pytest.mark.skip(reason="Takes a long time")
def test_route():
    if directory:
        import matplotlib.pyplot as plt
    nodes, map_con, map_con_latlon, route, route_latlon = load_data()
    zoom_path = True
    # zoom_path = slice(2645, 2665)

    # if directory is not None:
    #     logger.debug("Plotting pre map ...")
    #     mm_viz.plot_map(map_con_latlon, path=route_latlon, use_osm=True,
    #                     show_lattice=False, show_labels=False, show_graph=False, zoom_path=zoom_path,
    #                     filename=str(directory / "test_newson_route.png"))
    #     logger.debug("... done")

    matcher = DistanceMatcher(map_con, min_prob_norm=0.001,
                              max_dist=200,
                              dist_noise=6, dist_noise_ne=12,
                              obs_noise=30, obs_noise_ne=150,
                              non_emitting_states=True)
    matcher.match(route[2657:2662])  # First location where some observations are missing
    # matcher.match(route[2770:2800])  # Observations are missing
    # matcher.match(route[2910:2950])  # Interesting point
    # matcher.match(route[2910:2929])  # Interesting point
    # matcher.match(route[6000:])
    path_pred = matcher.path_pred_onlynodes

    if directory:
        matcher.print_lattice_stats()
        logger.debug("Plotting post map ...")
        fig = plt.figure(figsize=(200, 200))
        ax = fig.get_axes()
        mm_viz.plot_map(map_con, matcher=matcher, use_osm=True, ax=ax,
                        show_lattice=False, show_labels=False, zoom_path=zoom_path,
                        show_matching=True, show_graph=False,
                        coord_trans=map_con.yx2latlon)
        plt.savefig(str(directory / "test_newson_route_matched.png"))
        plt.close(fig)
        logger.debug("... done")
        logger.debug("Best path:")
        for m in matcher.lattice_best:
            logger.debug(m)

    print(path_pred)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_route()
    test_route_slice1()
    # test_bug1()
