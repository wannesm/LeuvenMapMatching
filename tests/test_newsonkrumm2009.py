#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_path_newsonkrumm2009
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the data available at:
https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/

:author: Wannes Meert
:copyright: Copyright 2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import os
import sys
import logging
from pathlib import Path
import csv
from datetime import datetime
import leuvenmapmatching as mm
from leuvenmapmatching.map.gdfmap import GDFMap
import leuvenmapmatching.visualization as mm_viz
MYPY = False
if MYPY:
    from typing import List, Tuple


this_path = Path(os.path.realpath(__file__)).parent / "rsrc" / "newson_krumm_2009"
gps_data = this_path / "gps_data.txt"
ground_truth_route = this_path / "ground_truth_route.txt"
road_network = this_path / "road_network.txt"

directory = None


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
    return nodes


def parse_linestring(line):
    # type: (str) -> List[Tuple[float, float]]
    line = line[line.index("(") + 1:line.index(")")]
    latlons = []
    for latlon in line.split(", "):
        lat, lon = latlon.split(" ")
        latlons.append((float(lat), float(lon)))
    return latlons


def read_map(map_fn):
    map = GDFMap(use_latlon=True)
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
            map.add_node(nf, innernodes[0])
            map.add_node(nt, innernodes[-1])
            prev_node = nf
            assert(length < 1000)
            for idx, innernode in enumerate(innernodes[1:-1]):
                innernode_id = nf * 1000 + idx
                map.add_node(innernode_id, innernode)
                map.add_edge(prev_node, innernode_id)
                map.add_edge(innernode_id, prev_node)
                prev_node = innernode_id
            map.add_edge(prev_node, nt)
            map.add_edge(nt, prev_node)
    return map


def test_route():
    route = read_gps(gps_data)
    print(f"Route length = {len(route)}")
    print(route)
    nodes = read_nodes(ground_truth_route)
    map_con = read_map(road_network)
    mm_viz.plot_map(map_con, path=route, use_osm=True,
                    show_lattice=False, show_labels=False, show_graph=False, zoom_path=True,
                    filename=str(directory / "test_newson_route.png"))



if __name__ == "__main__":
    mm.matching.logger.setLevel(logging.DEBUG)
    mm.matching.logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    test_route()
