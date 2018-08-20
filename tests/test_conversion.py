#!/usr/bin/env python3
# encoding: utf-8
import sys
import logging
from datetime import datetime
import pytest
import math

import leuvenmapmatching as mm
from leuvenmapmatching.util.gpx import path_to_gpx
from leuvenmapmatching.util.projections import latlon2grs80
from leuvenmapmatching.util import dist_euclidean as de
from leuvenmapmatching.util import dist_latlon as dll


def test_path_to_gpx():
    path = [(i, i, datetime.fromtimestamp(i)) for i in range(0, 10)]
    gpx = path_to_gpx(path)

    assert len(path) == len(gpx.tracks[0].segments[0].points)
    assert path[0][0] == pytest.approx(gpx.tracks[0].segments[0].points[0].latitude)
    assert path[0][1] == pytest.approx(gpx.tracks[0].segments[0].points[0].longitude)
    assert path[0][2] == pytest.approx(gpx.tracks[0].segments[0].points[0].time)


def test_grs80():
    coordinates = [(4.67878, 50.864), (4.68054, 50.86381), (4.68098, 50.86332), (4.68129, 50.86303), (4.6817, 50.86284),
                   (4.68277, 50.86371), (4.68894, 50.86895), (4.69344, 50.86987), (4.69354, 50.86992),
                   (4.69427, 50.87157), (4.69643, 50.87315), (4.69768, 50.87552), (4.6997, 50.87828)]
    points = latlon2grs80(coordinates, lon_0=coordinates[0][0], lat_ts=coordinates[0][1])
    points = list(points)
    point = points[0]
    assert point[0] == pytest.approx(618139.9385518166)
    assert point[1] == pytest.approx(5636043.991970774)


def test_distance1():
    p1 = (38.898556, -77.037852)
    p2 = (38.897147, -77.043934)
    d = dll.distance(p1, p2)
    assert d == pytest.approx(549.1557912048178), f"Got: {d}"


def test_distance2():
    o_p1 = (6007539.987516373, -13607675.997610645)
    m_p1 = (6007518.475594072, -13607641.049711559)
    m_p2 = (6007576.295597112, -13607713.306589901)
    dist, proj_m, t_m = de.distance_point_to_segment(o_p1, m_p1, m_p2)
    assert dist == pytest.approx(5.038773480896327), f"dist = {dist}"
    assert t_m == pytest.approx(0.4400926470800718), f"t_m = {t_m}"


def test_bearing1():
    lat1, lon1 = math.radians(38.898556), math.radians(-77.037852)
    lat2, lon2 = math.radians(38.897147), math.radians(-77.043934)
    b = dll.bearing_radians(lat1, lon1, lat2, lon2)
    b = math.degrees(b)
    # assert b == pytest.approx(253.42138889), f"Got: {b}"
    assert b == pytest.approx(-106.5748183426045), f"Got: {b}"


def test_destination1():
    lat1, lon1 = math.radians(53.32055556), math.radians(1.72972222)
    bearing = math.radians(96.02166667)
    dist = 124800
    lat2, lon2 = dll.destination_radians(lat1, lon1, bearing, dist)
    lat2, lon2 = (math.degrees(lat2), math.degrees(lon2))
    print(lat2, lon2)
    assert (lat2, lon2) == (53.188269553709034, 3.592721390871882), f"Got ({lat2}, {lon2})"


if __name__ == "__main__":
    # mm.matching.logger.setLevel(logging.INFO)
    mm.matching.logger.setLevel(logging.DEBUG)
    mm.matching.logger.addHandler(logging.StreamHandler(sys.stdout))
    # test_path_to_gpx()
    # test_grs80()
    # test_distance1()
    # test_bearing1()
    test_destination1()
