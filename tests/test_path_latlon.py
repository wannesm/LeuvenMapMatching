#!/usr/bin/env python3
# encoding: utf-8
"""
tests.test_path_latlon
~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import sys
import os
import logging
from pathlib import Path
import osmread
import pytest
import leuvenmapmatching as mm
import leuvenmapmatching.visualization as mm_viz
from leuvenmapmatching.util.gpx import gpx_to_path
from leuvenmapmatching.util.dist_latlon import interpolate_path

this_path = Path(os.path.realpath(__file__)).parent / "rsrc"
osm_fn = this_path / "osm_downloaded.xml"
osm2_fn = this_path / "osm_downloaded2.xml"
track_fn = this_path / "route.gpx"
track2_fn = this_path / "route2.gpx"
directory = None


def prepare_files(verbose=False):
    if not osm_fn.exists():
        if verbose:
            print("Downloading {}".format(osm_fn))
        import requests
        url = 'http://overpass-api.de/api/map?bbox=4.694933,50.870047,4.709256000000001,50.879628'
        r = requests.get(url, stream=True)
        with osm_fn.open('wb') as ofile:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    ofile.write(chunk)
    if not osm2_fn.exists():
        if verbose:
            print("Downloading {}".format(osm2_fn))
        import requests
        url = 'http://overpass-api.de/api/map?bbox=4.6997666,50.8684188,4.7052813,50.8731718'
        r = requests.get(url, stream=True)
        with osm2_fn.open('wb') as ofile:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    ofile.write(chunk)


def create_map():
    map_con = mm.map.InMemMap(use_latlon=True)
    cnt = 0
    for entity in osmread.parse_file(str(osm_fn)):
        if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
            for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                map_con.add_edge(node_a, None, node_b, None)
                # Some roads are one-way. We'll add both directions.
                map_con.add_edge(node_b, None, node_a, None)
        if isinstance(entity, osmread.Node):
            map_con.add_node(entity.id, (entity.lat, entity.lon))
    map_con.purge()
    return map_con


def create_map2():
    map_con = mm.map.InMemMap(use_latlon=True)
    cnt = 0
    for entity in osmread.parse_file(str(osm2_fn)):
        if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
            for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                map_con.add_edge(node_a, None, node_b, None)
                # Some roads are one-way. We'll add both directions.
                map_con.add_edge(node_b, None, node_a, None)
        if isinstance(entity, osmread.Node):
            map_con.add_node(entity.id, (entity.lat, entity.lon))
    map_con.purge()
    return map_con


def plot_path(max_nodes=None):
    prepare_files()
    track = gpx_to_path(track2_fn)
    if max_nodes is not None:
        track = track[:max_nodes]
    path = [(lat, lon) for lat, lon, _ in track]
    print(path)
    map_con = create_map2()

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(100, 100))
    # mm_viz.plot_map(map_con, path=path, nodes=nodes, z=16, use_osm=True, tilesize=512)
    mm_viz.plot_map(map_con, path=path, z=17, use_osm=True, zoom_path=False,
                    filename=str(directory / "test_path_latlon_map.png"))


@pytest.mark.skip(reason="Ignore LatLon for now")
def test_path1():
    prepare_files()
    track = gpx_to_path(track_fn)
    track = track[:3]
    track_int = interpolate_path(track, 5)
    map_con = create_map()

    matcher = mm.matching.Matcher(map_con, max_dist=50, obs_noise=50, min_prob_norm=0.1)
    nodes, last_idx = matcher.match(track_int, unique=False)
    if len(nodes) < len(track_int):
        raise Exception(f"Could not find a match for the full path. Last index = {last_idx}")
    if directory:
        matcher.print_lattice_stats()
        path = [(lat, lon) for lat, lon, _ in track]
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(100, 100))
        # mm_viz.plot_map(map_con, path=path, nodes=nodes, z=16, use_osm=True, tilesize=512)
        mm_viz.plot_map(map_con, z=16, use_osm=True,
                        filename=str(directory / "test_path_latlon_path1.png"))


@pytest.mark.skip(reason="Ignore LatLon for now")
def test_path2():
    prepare_files()
    track = gpx_to_path(track2_fn)
    # track = track[:3]
    # track = mm.util.interpolate_path(track, 5)
    map_con = create_map2()
    matcher = mm.matching.Matcher(map_con, max_dist=100, max_dist_init=20,
                                  obs_noise=50, min_prob_norm=0.1,
                                  max_lattice_width=5,
                                  non_emitting_states=True)
    nodes, last_idx = matcher.match(track, unique=False)
    if len(nodes) < len(track):
        raise Exception(f"Could not find a match for the full path. Last index = {last_idx}")
    if directory:
        matcher.print_lattice_stats()
        path = [(lat, lon) for lat, lon, _ in track]
        mm_viz.plot_map(map_con, matcher=matcher, nodes=nodes, path=path, z=17, use_osm=True,
                        show_lattice=True, filename=str(directory / "test_path_latlon_path2.png"))


@pytest.mark.skip(reason="Ignore LatLon for now")
def test_path2_onlyedges():
    prepare_files()
    track = gpx_to_path(track2_fn)
    # track = track[:3]
    # track = mm.util.interpolate_path(track, 5)
    map_con = create_map2()
    matcher = mm.matching.Matcher(map_con, max_dist=100, max_dist_init=20,
                                  obs_noise=50, min_prob_norm=0.1,
                                  max_lattice_width=5,
                                  non_emitting_states=True, only_edges=True)
    nodes, last_idx = matcher.match(track, unique=False)
    if len(nodes) < len(track):
        raise Exception(f"Could not find a match for the full path. Last index = {last_idx}")
    if directory:
        matcher.print_lattice_stats()
        path = [(lat, lon) for lat, lon, _ in track]
        mm_viz.plot_map(map_con, matcher=matcher, nodes=nodes, path=path, z=17, use_osm=True,
                        show_lattice=True, filename=str(directory / "test_path_latlon_path2.png"))


if __name__ == "__main__":
    mm.matching.logger.setLevel(logging.DEBUG)
    mm.matching.logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_path1()
    # plot_path(max_nodes=None)
    # test_path2()
    test_path2_onlyedges()
