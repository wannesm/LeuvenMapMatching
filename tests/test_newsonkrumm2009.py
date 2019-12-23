#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_path_newsonkrumm2009
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the data available at:
https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/

Notes:

* There is a 'bug' in the map available from the website.
  Multiple segments (streets) in the map are not connected but have overlapping, but
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
from itertools import product
import pytest
import leuvenmapmatching as mm
from leuvenmapmatching.matcher import base
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.sqlite import SqliteMap
import leuvenmapmatching.visualization as mm_viz
MYPY = False
if MYPY:
    from typing import List, Tuple


logger = mm.logger
this_path = Path(os.path.realpath(__file__)).parent / "rsrc" / "newson_krumm_2009"
gps_data = this_path / "gps_data.txt"
gps_data_pkl = gps_data.with_suffix(".pkl")
ground_truth_route = this_path / "ground_truth_route.txt"
road_network = this_path / "road_network.txt"
road_network_db = road_network.with_suffix(".sqlite")

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


def read_paths(paths_fn):
    paths = []
    with paths_fn.open("r") as paths_f:
        reader = csv.reader(paths_f, delimiter='\t')
        next(reader)
        for row in reader:
            pathid, trav = row[:2]
            pathid = int(pathid)
            trav = int(trav)
            paths.append((pathid, trav))
    logger.debug(f"Read correct trace of {len(paths)} nodes")
    return paths


def parse_linestring(line):
    # type: (str) -> List[Tuple[float, float]]
    line = line[line.index("(") + 1:line.index(")")]
    latlons = []
    for lonlat in line.split(", "):
        lon, lat = lonlat.split(" ")
        latlons.append((float(lat), float(lon)))
    return latlons


def read_map(map_fn):
    logger.debug(f"Reading map ...")
    mmap = SqliteMap("road_network", use_latlon=True, dir=this_path)
    node_cnt = 0
    edge_cnt = 0
    # new_node_id = 1000000000000
    new_node_id = 1
    with map_fn.open("r") as map_f:
        reader = csv.reader(map_f, delimiter='\t')
        next(reader)
        for row in reader:
            eid, nf, nt, twoway, speed, length, innernodes = row
            eid = int(eid)
            nf = int(nf)
            nt = int(nt)
            length = int(length)
            twoway = int(twoway)
            speed = float(speed)
            if twoway == 0:
                twoway = False
            elif twoway == 1:
                twoway = True
            else:
                raise Exception(f"Unknown value for twoway: {twoway}")
            innernodes = parse_linestring(innernodes)
            # Add nodes to map
            mmap.add_node(nf, innernodes[0], ignore_doubles=True, no_index=True, no_commit=True)
            mmap.add_node(nt, innernodes[-1], ignore_doubles=True, no_index=True, no_commit=True)
            node_cnt += 2
            prev_node = nf
            assert(length < 1000)
            idx = 1
            for innernode in innernodes[1:-1]:
                # innernode_id = nf * 1000 + idx
                innernode_id = new_node_id
                new_node_id += 1
                mmap.add_node(innernode_id, innernode, no_index=True, no_commit=True)  # Should not be double
                node_cnt += 1
                mmap.add_edge(prev_node, innernode_id, speed=speed, edge_type=0,
                              path=eid, pathnum=idx, no_index=True, no_commit=True)
                edge_cnt += 1
                if twoway:
                    mmap.add_edge(innernode_id, prev_node, speed=speed, edge_type=0,
                                  path=eid, pathnum=-idx, no_index=True, no_commit=True)
                    edge_cnt += 1
                prev_node = innernode_id
                idx += 1
            mmap.add_edge(prev_node, nt, speed=speed, edge_type=0,
                          path=eid, pathnum=idx, no_index=True, no_commit=True)
            edge_cnt += 1
            if twoway:
                mmap.add_edge(nt, prev_node, speed=speed, edge_type=0,
                              path=eid, pathnum=-idx, no_index=True, no_commit=True)
                edge_cnt += 1
            if node_cnt % 100000 == 0:
                mmap.db.commit()
    logger.debug(f"... done: {node_cnt} nodes and {edge_cnt} edges")
    mmap.reindex_nodes()
    mmap.reindex_edges()
    assert(new_node_id < 100000000000)
    return mmap


def correct_map(mmap):
    """Add edges between nodes with degree > 2 that are on the exact same location.
    This ignore that with bridges, the roads might not be connected. But we need a correct
    because the dataset has a number of interrupted paths.
    """
    def correct_edge(labels):
        labels = [label for label in labels if label > 100000000000]
        logger.info(f"Add connections between {labels}")
        for l1, l2 in product(labels, repeat=2):
            mmap.add_edge(l1, l2, edge_type=1)
    mmap.find_duplicates(func=correct_edge)


def load_data():
    max_route_length = None  # 200

    # Paths
    paths = read_paths(ground_truth_route)

    # Map
    if road_network_db.exists():
        map_con = SqliteMap.from_file(road_network_db)
        logger.debug(f"Read road network from db file {road_network_db} ({map_con.size()} nodes)")
    else:
        map_con = read_map(road_network)
        correct_map(map_con)
        logger.debug(f"Create road network to db file {map_con.db_fn} ({map_con.size()} nodes)")

    # Route
    if gps_data_pkl.exists():
        with gps_data_pkl.open("rb") as ifile:
            route = pickle.load(ifile)
        logger.debug(f"Read gps route from file ({len(route)} points)")
    else:
        route = read_gps(gps_data)
        if max_route_length:
            route = route[:max_route_length]
        with gps_data_pkl.open("wb") as ofile:
            pickle.dump(route, ofile)
    route = [(lat, lon) for lat, lon, _ in route]

    return paths, map_con, route


def test_route_slice1():
    if directory:
        import matplotlib.pyplot as plt
    nodes, map_con, route = load_data()
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
    map_con = SqliteMap("map", use_latlon=True)
    map_con.add_nodes([
        (1, (47.590439915657, -122.238368690014)),
        (2, (47.5910192728043, -122.239519357681)),
        (3, (47.5913706421852, -122.240168452263))
    ])
    map_con.add_edges([
        (1, 2),
        (2, 3)
    ])
    path = [
        # (47.59043333, -122.2384167),
        (47.59058333, -122.2387),
        (47.59071667, -122.2389833),
        (47.59086667, -122.2392667),
        (47.59101667, -122.23955),
        (47.59115,    -122.2398333)
    ]
    path_sol = [(1, 2), (2, 3)]
    matcher = DistanceMatcher(map_con, min_prob_norm=0.001,
                              max_dist=200, obs_noise=4.07,
                              non_emitting_states=True)
    matcher.match(path, unique=True)
    path_pred = matcher.path_pred
    if directory:
        import matplotlib.pyplot as plt
        matcher.print_lattice_stats()
        logger.debug("Plotting post map ...")
        fig = plt.figure(figsize=(100, 100))
        ax = fig.get_axes()
        mm_viz.plot_map(map_con, matcher=matcher, use_osm=True, ax=ax,
                        show_lattice=False, show_labels=True, show_graph=True, zoom_path=True,
                        show_matching=True)
        plt.savefig(str(directory / "test_newson_bug1.png"))
        plt.close(fig)
        logger.debug("... done")
    assert path_pred == path_sol, f"Edges not equal:\n{path_pred}\n{path_sol}"


@pytest.mark.skip(reason="Takes a long time")
def test_route():
    if directory:
        import matplotlib.pyplot as plt
    else:
        plt = None
    paths, map_con, route = load_data()
    route = [(lat, lon) for lat, lon, _ in route]
    zoom_path = True
    # zoom_path = slice(2645, 2665)
    slice_route = None
    # slice_route = slice(650, 750)
    # slice_route = slice(2657, 2662)  # First location where some observations are missing
    # slice_route = slice(2770, 2800)  # Observations are missing
    # slice_route = slice(2910, 2950)  # Interesting point
    # slice_route = slice(2910, 2929)  # Interesting point
    # slice_route = slice(6825, 6833)  # Outlier observation

    # if directory is not None:
    #     logger.debug("Plotting pre map ...")
    #     mm_viz.plot_map(map_con_latlon, path=route_latlon, use_osm=True,
    #                     show_lattice=False, show_labels=False, show_graph=False, zoom_path=zoom_path,
    #                     filename=str(directory / "test_newson_route.png"))
    #     logger.debug("... done")

    matcher = DistanceMatcher(map_con, min_prob_norm=0.0001,
                              max_dist=200,
                              dist_noise=15, dist_noise_ne=30,
                              obs_noise=30, obs_noise_ne=150,
                              non_emitting_states=True)

    if slice_route is None:
        pkl_fn = this_path / "nodes_pred.pkl"
        if pkl_fn.exists():
            with pkl_fn.open("rb") as pkl_file:
                logger.debug(f"Reading predicted nodes from pkl file")
                route_nodes = pickle.load(pkl_file)
        else:
            matcher.match(route)
            route_nodes = matcher.path_pred_onlynodes
            with pkl_fn.open("wb") as pkl_file:
                pickle.dump(route_nodes, pkl_file)
        from leuvenmapmatching.util.evaluation import route_mismatch_factor
        print(route_nodes[:10])
        # route_edges = map_con.nodes_to_paths(route_nodes)
        # print(route_edges[:10])
        grnd_paths, _ = zip(*paths)
        print(grnd_paths[:10])
        route_paths = map_con.nodes_to_paths(route_nodes)
        print(route_paths[:10])

        logger.debug(f"Compute route mismatch factor")
        factor, cnt_matches, cnt_mismatches, total_length, mismatches = \
            route_mismatch_factor(map_con, route_paths, grnd_paths,window=None, keep_mismatches=True)
        logger.debug(f"factor = {factor}, "
                     f"cnt_matches = {cnt_matches}/{cnt_mismatches} of {len(grnd_paths)}/{len(route_paths)}, "
                     f"total_length = {total_length}\n"
                     f"mismatches = " + " | ".join(str(v) for v in mismatches))
    else:
        _, last_idx = matcher.match(route[slice_route])
        logger.debug(f"Last index = {last_idx}")

    # matcher.match(route[2657:2662])  # First location where some observations are missing
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
                        show_lattice=False, show_labels=True, zoom_path=zoom_path,
                        show_matching=True, show_graph=False)
        plt.savefig(str(directory / "test_newson_route_matched.png"))
        plt.close(fig)
        logger.debug("... done")
        logger.debug("Best path:")
        for m in matcher.lattice_best:
            logger.debug(m)

    print(path_pred)


@pytest.mark.skip(reason="Takes a too long")
def test_bug2():
    from leuvenmapmatching.util.openstreetmap import locations_to_map
    map_con = SqliteMap("map", use_latlon=True, dir=directory)
    path = [
        (50.87205, 4.66089), (50.874550000000006, 4.672980000000001), (50.87538000000001, 4.67698),
        (50.875800000000005, 4.6787600000000005), (50.876520000000006, 4.6818), (50.87688000000001, 4.683280000000001),
        (50.87814, 4.68733), (50.87832, 4.68778), (50.87879, 4.68851), (50.87903000000001, 4.68895),
        (50.879560000000005, 4.689170000000001), (50.87946, 4.6900900000000005),
        (50.879290000000005, 4.6909600000000005), (50.87906, 4.6921800000000005), (50.87935, 4.6924),
        (50.879720000000006, 4.69275), (50.88002, 4.6930700000000005), (50.880430000000004, 4.693440000000001),
        (50.880660000000006, 4.69357), (50.880660000000006, 4.6936100000000005), (50.88058, 4.694640000000001),
        (50.88055000000001, 4.69491), (50.88036, 4.696160000000001), (50.88009, 4.697550000000001),
        (50.87986, 4.6982800000000005), (50.879720000000006, 4.698790000000001), (50.87948, 4.699730000000001),
        (50.87914000000001, 4.6996400000000005), (50.87894000000001, 4.6995000000000005),
        (50.878800000000005, 4.699350000000001), (50.8785, 4.6991000000000005), (50.87841, 4.6990300000000005)
    ]
    locations_to_map(path, map_con, filename=directory / "osm.xml")
    path_sol = [(5777282112, 2633552218), (2633552218, 5777282111), (5777282111, 5777282110), (5777282110, 1642021707),
                (1642021707, 71361087), (71361087, 71364203), (71364203, 1151697757), (1151697757, 1647339017),
                (1647339017, 1647339030), (1647339030, 2058510349), (2058510349, 2633552212), (2633552212, 1380538577),
                (1380538577, 1439572271), (1439572271, 836434313), (836434313, 2633771041), (2633771041, 5042874484),
                (5042874484, 5042874485), (5042874485, 2518922583), (2518922583, 2659762546), (2659762546, 5777282063),
                (5777282063, 2633771037), (2633771037, 2633771035), (2633771035, 2633771033), (2633771033, 1151668705),
                (1151668705, 2633771094), (2633771094, 1151668722), (1151668722, 1151668724), (1151668724, 5543948222),
                (5543948222, 2058481517), (2058481517, 16933576), (16933576, 5543948221), (5543948221, 2518923620),
                (2518923620, 5543948020), (5543948020, 5543948019), (5543948019, 18635886), (18635886, 18635887),
                (18635887, 1036909153), (1036909153, 2658942230), (2658942230, 1001099975), (1001099975, 16933574),
                (16933574, 1125604152), (1125604152, 5543948238), (5543948238, 1125604150), (1125604150, 1125604148),
                (1125604148, 2634195334), (2634195334, 2087854243), (2087854243, 5543948237), (5543948237, 160226603),
                (160226603, 180130266), (180130266, 5543948227), (5543948227, 5543948226), (5543948226, 1195681902),
                (1195681902, 101135392), (101135392, 2606704673), (2606704673, 18635977), (18635977, 1026111708),
                (1026111708, 1026111631), (1026111631, 16571375), (16571375, 2000680621), (2000680621, 999580042),
                (999580042, 16571370), (16571370, 2000680620), (2000680620, 5078692402), (5078692402, 5543948008),
                (5543948008, 16571371), (16571371, 999579936), (999579936, 2639836143), (2639836143, 5543948014),
                (5543948014, 5222992316), (5222992316, 30251323), (30251323, 159701080), (159701080, 3173217124),
                (3173217124, 1165209673), (1165209673, 1380538689), (1380538689, 2878334668), (2878334668, 2871137399),
                (2871137399, 2876902981), (2876902981, 2873624508), (2873624508, 2873624509), (2873624509, 2899666507),
                (2899666507, 2899666518), (2899666518, 2899666513), (2899666513, 2903073945), (2903073945, 2903073951),
                (2903073951, 1380538681), (1380538681, 2914810627), (2914810627, 2914810618), (2914810618, 2914810607),
                (2914810607, 2914810604), (2914810604, 2914810483), (2914810483, 2914810462), (2914810462, 2914810464),
                (2914810464, 1312433523), (1312433523, 20918594), (20918594, 2634267817), (2634267817, 2967425445),
                (2967425445, 3201523879), (3201523879, 157217466), (157217466, 2963305939), (2963305939, 3201523877),
                (3201523877, 3889275909), (3889275909, 3889275897), (3889275897, 157255077), (157255077, 30251882),
                (30251882, 157245624), (157245624, 1150903673), (1150903673, 4504936404)]
    matcher = DistanceMatcher(map_con, min_prob_norm=0.001,
                              max_dist=200, obs_noise=4.07,
                              non_emitting_states=True)
    nodes, idx = matcher.match(path, unique=True)
    path_pred = matcher.path_pred
    if directory:
        import matplotlib.pyplot as plt
        matcher.print_lattice_stats()
        logger.debug("Plotting post map ...")
        fig = plt.figure(figsize=(100, 100))
        ax = fig.get_axes()
        mm_viz.plot_map(map_con, matcher=matcher, use_osm=True, ax=ax,
                        show_lattice=False, show_labels=True, show_graph=False, zoom_path=True,
                        show_matching=True)
        plt.savefig(str(directory / "test_newson_bug1.png"))
        plt.close(fig)
        logger.debug("... done")
    assert path_pred == path_sol, f"Edges not equal:\n{path_pred}\n{path_sol}"


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_route()
    # test_route_slice1()
    # test_bug1()
    test_bug2()
