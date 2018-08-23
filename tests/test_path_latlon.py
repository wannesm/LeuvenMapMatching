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
from functools import partial
import leuvenmapmatching as mm
import leuvenmapmatching.visualization as mm_viz
from leuvenmapmatching.util.gpx import gpx_to_path
from leuvenmapmatching.util.dist_latlon import interpolate_path
from leuvenmapmatching.util.projections import latlon2grs80
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.simple import SimpleMatcher

logger = mm.logger
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
    map_con = InMemMap("map", use_latlon=True)
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


def create_map2(convert_latlon=None):
    use_latlon = True if convert_latlon is None else False
    map_con = InMemMap("map2", use_latlon=use_latlon)
    cnt = 0
    for entity in osmread.parse_file(str(osm2_fn)):
        if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
            for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                map_con.add_edge(node_a, node_b)
                # Some roads are one-way. We'll add both directions.
                map_con.add_edge(node_b, node_a)
        if isinstance(entity, osmread.Node):
            if convert_latlon is None:
                lat = entity.lat
                lon = entity.lon
            else:
                lat, lon = list(convert_latlon([(entity.lat, entity.lon)]))[0]
            map_con.add_node(entity.id, (lat, lon))
    map_con.purge()
    return map_con


def plot_path(max_nodes=None):
    prepare_files()
    track = gpx_to_path(track2_fn)
    if max_nodes is not None:
        track = track[:max_nodes]
    path = [(lat, lon) for lat, lon, _ in track]
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

    matcher = SimpleMatcher(map_con, max_dist=50, obs_noise=50, min_prob_norm=0.1)
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


def test_path2_proj_e():
    prepare_files()
    track = [p[:2] for p in gpx_to_path(track2_fn)]
    proj = partial(latlon2grs80, lon_0=track[0][1], lat_ts=track[0][0])
    track = list(proj(track))
    # track = track[:3]
    # track = mm.util.interpolate_path(track, 5)
    map_con = create_map2(convert_latlon=proj)
    matcher = SimpleMatcher(map_con, max_dist=300, max_dist_init=25,
                            obs_noise=75, obs_noise_ne=125, min_prob_norm=0.1,
                            max_lattice_width=5, only_edges=True,
                            non_emitting_states=True)
    path, last_idx = matcher.match(track, unique=False)
    nodes = matcher.path_pred_onlynodes
    if len(path) < len(track):
        raise Exception(f"Could not find a match for the full path. Last index = {last_idx}")
    if directory:
        matcher.print_lattice_stats()
        mm_viz.plot_map(map_con, matcher=matcher, path=track, use_osm=False,
                        show_lattice=True, show_matching=True, show_labels=3,
                        filename=str(directory / "test_path_latlon_path2_proj_e.png"))
    # print(nodes)
    nodes_sol = [5435850758, 2634474831, 1096512242, 3051083902, 1096512239, 1096512241, 1096512240,
                 159654664, 1096508373, 1096508381, 16483859, 1096508369, 159654663, 1096508363,
                 16483862, 3051083898, 16526535, 3060597381, 3060515059, 16526534, 3060515058,
                 16483865, 16483866, 3060725817, 611867918, 16526532, 1274158119, 16526540,
                 3060597377, 16526541, 16424220, 1233373340, 613125597, 1076057753]
    assert nodes == nodes_sol, f"Nodes do not match: {nodes}"


def test_path2_proj_ne():
    prepare_files()
    track = [p[:2] for p in gpx_to_path(track2_fn)]
    proj = partial(latlon2grs80, lon_0=track[0][1], lat_ts=track[0][0])
    track = list(proj(track))
    # track = track[:3]
    # track = mm.util.interpolate_path(track, 5)
    map_con = create_map2(convert_latlon=proj)
    matcher = SimpleMatcher(map_con, max_dist=300, max_dist_init=25,
                            obs_noise=75, obs_noise_ne=125, min_prob_norm=0.1,
                            max_lattice_width=5, only_edges=False,
                            non_emitting_states=True)
    path, last_idx = matcher.match(track, unique=False)
    nodes = matcher.path_pred_onlynodes
    if len(path) < len(track):
        raise Exception(f"Could not find a match for the full path. Last index = {last_idx}")
    if directory:
        matcher.print_lattice_stats()
        mm_viz.plot_map(map_con, matcher=matcher, path=track, use_osm=False,
                        show_lattice=True, show_matching=True, show_labels=3,
                        filename=str(directory / "test_path_latlon_path2_proj_ne.png"))
    # print(nodes)
    nodes_sol = [5435850758, 2634474829, 5435850755, 1096512241, 1096512240, 1096508366, 1096508372,
                 16483861, 1096508360, 159656075, 1096508382, 16483862, 3051083898, 16526535,
                 3060597381, 3060515059, 16526534, 16526532, 1274158119, 16526540, 3060597377,
                 16526541, 16424220, 16483842, 16424220, 1233373340, 613125597, 1076057753]
    assert nodes == nodes_sol, f"Nodes do not match: {nodes}"


@pytest.mark.skip(reason="Ignore LatLon for now")
def test_path2():
    prepare_files()
    track = gpx_to_path(track2_fn)
    # track = track[:3]
    # track = mm.util.interpolate_path(track, 5)
    map_con = create_map2()
    matcher = SimpleMatcher(map_con, max_dist=100, max_dist_init=20,
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
                        show_lattice=True, filename=str(directory / "test_path_latlon_path2_ne.png"))


@pytest.mark.skip(reason="Ignore LatLon for now")
def test_path2_onlyedges():
    prepare_files()
    track = gpx_to_path(track2_fn)
    # track = track[:3]
    # track = mm.util.interpolate_path(track, 5)
    map_con = create_map2()
    matcher = SimpleMatcher(map_con, max_dist=100, max_dist_init=20,
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
                        show_lattice=True, filename=str(directory / "test_path_latlon_path2_e.png"))


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_path1()
    # plot_path(max_nodes=None)
    # test_path2_proj_e()
    test_path2_proj_ne()
    # test_path2()
    # test_path2_onlyedges()
