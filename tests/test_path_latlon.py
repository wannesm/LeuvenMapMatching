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
from leuvenmapmatching.util.openstreetmap import create_map_from_xml, download_map_xml
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher

logger = mm.logger
this_path = Path(os.path.realpath(__file__)).parent / "rsrc"
osm_fn = this_path / "osm_downloaded.xml"
osm2_fn = this_path / "osm_downloaded2.xml"
osm3_fn = this_path / "osm_downloaded3.xml"
track_fn = this_path / "route.gpx"  # http://users.telenet.be/meirenwi/Leuven%20Stadswandeling%20-%205%20km%20RT.zip
track2_fn = this_path / "route2.gpx"
track3_fn = this_path / "route3.pgx"
directory = None


def prepare_files(verbose=False, force=False):
    download_map_xml(osm_fn, '4.694933,50.870047,4.709256000000001,50.879628', force=force, verbose=verbose)
    download_map_xml(osm2_fn, '4.6997666,50.8684188,4.7052813,50.8731718', force=force, verbose=verbose)
    download_map_xml(osm3_fn, '4.69049,50.86784,4.71604,50.88784', force=force, verbose=verbose)


def test_path1():
    prepare_files()
    track = gpx_to_path(track_fn)
    track = [loc[:2] for loc in track]
    track = track[:5]
    track_int = interpolate_path(track, 5)
    map_con = create_map_from_xml(osm_fn)

    matcher = DistanceMatcher(map_con, max_dist=50, obs_noise=50, min_prob_norm=0.1)
    states, last_idx = matcher.match(track_int)

    if directory:
        # matcher.print_lattice_stats()
        mm_viz.plot_map(map_con, matcher=matcher, use_osm=True,
                        zoom_path=True, show_graph=True,
                        filename=str(directory / "test_path_latlon_path1.png"))
    assert len(states) == len(track_int), f"Path ({len(track_int)}) not fully matched by best path ({len(states)}), " + \
                                          f"last index = {last_idx}"
    states_sol = [(2963305939, 249348325), (2963305939, 249348325), (2963305939, 249348325), (2963305939, 249348325),
                  (2963305939, 249348325), (2963305939, 249348325), (249348325, 1545679243), (249348325, 1545679243),
                  (1545679243, 3663115134), (1545679243, 3663115134), (1545679243, 3663115134),
                  (3663115134, 1545679251), (1545679251, 20910628), (1545679251, 20910628), (1545679251, 20910628),
                  (1545679251, 20910628), (20910628, 3663115130)]
    assert states == states_sol, f"Got states: {states}"


@pytest.mark.skip(reason="Takes a long time")
def test_path1_full():
    prepare_files()
    track = gpx_to_path(track_fn)
    track = [loc[:2] for loc in track]
    track_int = interpolate_path(track, 5)
    map_con = create_map_from_xml(osm_fn, include_footways=True, include_parking=True)

    matcher = DistanceMatcher(map_con, max_dist=50, obs_noise=50, min_prob_norm=0.1)
    states, last_idx = matcher.match(track_int)

    if directory:
        # matcher.print_lattice_stats()
        mm_viz.plot_map(map_con, matcher=matcher, use_osm=True,
                        zoom_path=True, show_graph=True,
                        filename=str(directory / "test_path_latlon_path1.png"))
    assert len(states) == len(track_int), f"Path ({len(track_int)}) not fully matched by best path ({len(states)}), " + \
                                          f"last index = {last_idx}"


def test_path2_proj():
    prepare_files()
    map_con_latlon = create_map_from_xml(osm2_fn)
    map_con = map_con_latlon.to_xy()
    track = [map_con.latlon2yx(p[0], p[1]) for p in gpx_to_path(track2_fn)]
    matcher = DistanceMatcher(map_con, max_dist=300, max_dist_init=25, min_prob_norm=0.0001,
                              non_emitting_length_factor=0.95,
                              obs_noise=50, obs_noise_ne=50,
                              dist_noise=50,
                              max_lattice_width=5,
                              non_emitting_states=True)
    states, last_idx = matcher.match(track, unique=False)
    nodes = matcher.path_pred_onlynodes
    if directory:
        matcher.print_lattice_stats()
        mm_viz.plot_map(map_con, matcher=matcher, path=track, use_osm=False,
                        show_graph=True, show_matching=True, show_labels=5,
                        filename=str(directory / "test_path_latlon_path2_proj.png"))
    nodes_sol = [2634474831, 1096512242, 3051083902, 1096512239, 1096512241, 1096512240, 1096508366, 1096508372,
                 16483861, 1096508360, 159656075, 1096508382, 16483862, 3051083898, 16526535, 3060597381, 3060515059,
                 16526534, 16526532, 1274158119, 16526540, 3060597377, 16526541, 16424220, 1233373340, 613125597,
                 1076057753]
    nodes_sol2 = [1096512242, 3051083902, 1096512239, 1096512241, 1096512240, 159654664, 1096508373, 1096508381,
                  16483859, 1096508369, 159654663, 1096508363, 16483862, 3051083898, 16526535, 3060597381, 3060515059,
                  16526534, 16526532, 611867918, 3060725817, 16483866, 3060725817, 611867918, 16526532, 1274158119,
                  16526540, 3060597377, 16526541, 16424220, 1233373340, 613125597, 1076057753]
    assert (nodes == nodes_sol) or (nodes == nodes_sol2), f"Nodes do not match: {nodes}"


def test_path2():
    prepare_files()
    map_con = create_map_from_xml(osm2_fn)
    track = [(p[0], p[1]) for p in gpx_to_path(track2_fn)]
    matcher = DistanceMatcher(map_con, max_dist=300, max_dist_init=25, min_prob_norm=0.0001,
                              non_emitting_length_factor=0.95,
                              obs_noise=50, obs_noise_ne=50,
                              dist_noise=50,
                              max_lattice_width=5,
                              non_emitting_states=True)
    states, last_idx = matcher.match(track, unique=False)
    nodes = matcher.path_pred_onlynodes
    if directory:
        mm_viz.plot_map(map_con, matcher=matcher, nodes=nodes, path=track, z=17, use_osm=True,
                        show_graph=True, show_matching=True,
                        filename=str(directory / "test_path_latlon_path2.png"))
    nodes_sol = [2634474831, 1096512242, 3051083902, 1096512239, 1096512241, 1096512240, 1096508366, 1096508372,
                 16483861, 3051083900, 16483864, 16483865, 3060515058, 16526534, 16526532, 1274158119, 16526540,
                 3060597377, 16526541, 16424220, 1233373340, 613125597, 1076057753]
    nodes_sol2 = [2634474831, 1096512242, 3051083902, 1096512239, 1096512241, 1096512240, 159654664, 1096508373,
                  1096508381, 16483859, 1096508369, 159654663, 1096508363, 16483862, 3051083898, 16526535, 3060597381,
                  3060515059, 16526534, 16526532, 1274158119, 16526540, 3060597377, 16526541, 16424220, 1233373340,
                  613125597, 1076057753]

    assert (nodes == nodes_sol) or (nodes == nodes_sol2), f"Nodes do not match: {nodes}"


def test_path3():
    prepare_files()
    track = [(50.87881, 4.698930000000001), (50.87899, 4.69836), (50.87905000000001, 4.698110000000001),
             (50.879000000000005, 4.69793), (50.87903000000001, 4.69766), (50.87906, 4.697500000000001),
             (50.87908, 4.6973), (50.879110000000004, 4.69665), (50.87854, 4.696420000000001),
             (50.878440000000005, 4.696330000000001), (50.878370000000004, 4.696140000000001),
             (50.8783, 4.69578), (50.87832, 4.69543), (50.87767, 4.695530000000001),
             (50.87763, 4.695080000000001), (50.87758, 4.6948300000000005), (50.877480000000006, 4.69395),
             (50.877500000000005, 4.693700000000001), (50.877520000000004, 4.69343),
             (50.877610000000004, 4.692670000000001), (50.87776, 4.6917800000000005),
             (50.87783, 4.69141), (50.87744000000001, 4.6908900000000004), (50.87736, 4.690790000000001),
             (50.877300000000005, 4.69078), (50.876650000000005, 4.6907000000000005),
             (50.87597, 4.69066), (50.875820000000004, 4.69068), (50.87561, 4.6907700000000006),
             (50.874430000000004, 4.69136), (50.874210000000005, 4.691490000000001), (50.87413, 4.69151),
             (50.87406000000001, 4.69151), (50.87397000000001, 4.69148), (50.87346, 4.6913800000000005),
             (50.87279, 4.691260000000001), (50.872490000000006, 4.69115), (50.87259, 4.6908900000000004),
             (50.87225, 4.690650000000001), (50.872080000000004, 4.6904900000000005),
             (50.871550000000006, 4.69125), (50.87097000000001, 4.69216), (50.87033, 4.69324),
             (50.87017, 4.6935400000000005), (50.87012000000001, 4.69373), (50.86997, 4.69406),
             (50.86981, 4.694520000000001), (50.86943, 4.69585), (50.868970000000004, 4.697500000000001),
             (50.868770000000005, 4.698130000000001), (50.86863, 4.6985), (50.86844000000001, 4.69899),
             (50.868140000000004, 4.69977), (50.86802, 4.70023), (50.867920000000005, 4.70078),
             (50.86787, 4.701180000000001), (50.86784, 4.70195), (50.86786000000001, 4.702310000000001),
             (50.86791, 4.702870000000001), (50.86836, 4.7052700000000005), (50.86863, 4.7064900000000005),
             (50.86880000000001, 4.707210000000001), (50.869220000000006, 4.708410000000001),
             (50.869400000000006, 4.70891), (50.86959, 4.709350000000001), (50.86995, 4.71004), (50.87006, 4.71021),
             (50.870900000000006, 4.7112300000000005), (50.872260000000004, 4.712890000000001), (50.87308, 4.71389),
             (50.873430000000006, 4.714300000000001), (50.873560000000005, 4.71441),
             (50.873740000000005, 4.714530000000001), (50.874280000000006, 4.714740000000001),
             (50.876250000000006, 4.71544), (50.876490000000004, 4.7155700000000005),
             (50.876900000000006, 4.7158500000000005), (50.87709, 4.71598), (50.877190000000006, 4.716010000000001),
             (50.87751, 4.7160400000000005), (50.87782000000001, 4.7160400000000005), (50.87832, 4.71591),
             (50.87894000000001, 4.71567), (50.87975, 4.71536), (50.88004, 4.71525), (50.8804, 4.715070000000001),
             (50.88163, 4.71452), (50.881750000000004, 4.71447), (50.8819, 4.714390000000001),
             (50.882200000000005, 4.71415), (50.882470000000005, 4.7138800000000005),
             (50.883480000000006, 4.7127300000000005), (50.88552000000001, 4.710470000000001),
             (50.88624, 4.70966), (50.88635000000001, 4.7096100000000005), (50.886520000000004, 4.709580000000001),
             (50.88664000000001, 4.7095400000000005), (50.886750000000006, 4.709280000000001),
             (50.88684000000001, 4.70906), (50.886970000000005, 4.70898), (50.88705, 4.70887), (50.88714, 4.70868),
             (50.88743, 4.7079), (50.887840000000004, 4.7069), (50.88776000000001, 4.70687),
             (50.88765, 4.706790000000001), (50.887100000000004, 4.70627), (50.88702000000001, 4.70619),
             (50.886950000000006, 4.706040000000001), (50.886950000000006, 4.7058800000000005),
             (50.886970000000005, 4.705620000000001), (50.88711000000001, 4.70417), (50.88720000000001, 4.70324),
             (50.88723, 4.7027600000000005), (50.88709000000001, 4.70253), (50.886480000000006, 4.70148),
             (50.88636, 4.70131), (50.886050000000004, 4.70101), (50.88593, 4.70092),
             (50.885810000000006, 4.700880000000001), (50.88539, 4.7008600000000005), (50.88497, 4.70082),
             (50.88436, 4.70089), (50.88398, 4.70094), (50.883250000000004, 4.7010700000000005),
             (50.88271, 4.701160000000001), (50.88136, 4.70159), (50.881130000000006, 4.701790000000001),
             (50.880930000000006, 4.7020100000000005), (50.88078, 4.70223), (50.88046000000001, 4.70146),
             (50.88015000000001, 4.70101), (50.880030000000005, 4.700880000000001), (50.87997000000001, 4.70078),
             (50.879900000000006, 4.70061), (50.87984, 4.70052), (50.879960000000004, 4.70026)]
    track = track[:30]
    map_con = create_map_from_xml(osm3_fn)

    matcher = DistanceMatcher(map_con,
                              max_dist_init=30, max_dist=50, min_prob_norm=0.1,
                              obs_noise=10, obs_noise_ne=20, dist_noise=10,
                              non_emitting_states=True)
    states, last_idx = matcher.match(track)

    if directory:
        # matcher.print_lattice_stats()
        mm_viz.plot_map(map_con, matcher=matcher, use_osm=True,
                        zoom_path=True, show_graph=False, show_matching=True,
                        filename=str(directory / "test_path_latlon_path3.png"))
    nodes = matcher.path_pred_onlynodes
    nodes_sol = [3906576303, 1150903750, 4506996820, 4506996819, 4506996798, 3906576457, 130147477, 3906576346,
                 231974072, 231974123, 1180606706, 19792164, 19792172, 1180606683, 1180606709, 5236409057,
                 19792169, 5236409056, 180241961, 180241975, 4506996259, 19792156, 5236409048, 180241625,
                 180241638, 231953030, 241928030, 241928031, 83796665, 231953028, 1125556965, 1380538625,
                 1824115892, 4909655515, 16571387, 16737662, 16571388, 179425214, 3705540990, 4567021046]
    assert nodes == nodes_sol, f"Nodes do not match: {nodes}"


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_path1()
    # test_path1_full()
    # test_path2_proj()
    test_path2()
    # test_path3()
