#!/usr/bin/env python3
# encoding: utf-8
import sys
import logging
from datetime import datetime
import pytest
import math
import os
from pathlib import Path
import itertools

import leuvenmapmatching as mm
from leuvenmapmatching.util import dist_euclidean as de
from leuvenmapmatching.util import dist_latlon as dll


directory = None


def test_path_to_gpx():
    from leuvenmapmatching.util.gpx import path_to_gpx
    path = [(i, i, datetime.fromtimestamp(i)) for i in range(0, 10)]
    gpx = path_to_gpx(path)

    assert len(path) == len(gpx.tracks[0].segments[0].points)
    assert path[0][0] == pytest.approx(gpx.tracks[0].segments[0].points[0].latitude)
    assert path[0][1] == pytest.approx(gpx.tracks[0].segments[0].points[0].longitude)
    assert path[0][2] == gpx.tracks[0].segments[0].points[0].time


def test_grs80():
    from leuvenmapmatching.util.projections import latlon2grs80
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


def test_distance_segment_to_segment1():
    f1 = (50.900393, 4.728607)
    f2 = (50.900389, 4.734047)
    t1 = (50.898538, 4.726107)
    t2 = (50.898176, 4.735463)
    d, pf, pt, u_f, u_t = dll.distance_segment_to_segment(f1, f2, t1, t2)
    if directory:
        plot_distance_segment_to_segment_latlon(f1, f2, t1, t2, pf, pt, "test_distance_segment_to_segment1")
    assert d == pytest.approx(216.60187728486514)
    assert pf == pytest.approx((50.900392999999994, 4.728607))
    assert pt == pytest.approx((50.898448650708666, 4.728418070396815))
    assert u_f == pytest.approx(0)
    assert u_t == pytest.approx(0.2470133466162735)


def test_distance_segment_to_segment2():
    f1 = (0, 0)
    f2 = (-0.43072496752146333, 381.4928613075559)
    t1 = (-206.26362055248765, -175.32538004745732)
    t2 = (-246.4746107556939, 480.8174213050763)
    d, pf, pt, u_f, u_t = de.distance_segment_to_segment(f1, f2, t1, t2)
    if directory:
        plot_distance_segment_to_segment_euc(f1, f2, t1, t2, pf, pt, "test_distance_segment_to_segment2")
    assert d == pytest.approx(216.60187728486514)
    assert pf == pytest.approx((0.0, 0.0))
    assert pt == pytest.approx((-216.1962718133358, -13.249350827191222))
    assert u_f == pytest.approx(0)
    assert u_t == pytest.approx(0.2470133466162735)


def test_distance_segment_to_segment3():
    f1 = (50.87205, 4.66089)
    f2 = (50.874550000000006, 4.672980000000001)
    t1 = (50.8740376, 4.6705204)
    t2 = (50.8741866999999, 4.67119980000001)
    d, pf, pt, u_f, u_t = dll.distance_segment_to_segment(f1, f2, t1, t2)
    if directory:
        plot_distance_segment_to_segment_latlon(f1, f2, t1, t2, pf, pt, "test_distance_segment_to_segment3")
    assert d == pytest.approx(0)
    assert pf == pytest.approx((50.87410572908839, 4.670830969750696))
    assert pt == pytest.approx((50.87410575464133, 4.670830955670548))
    assert u_f == pytest.approx(0.8222551304652699)
    assert u_t == pytest.approx(0.4571036354431931)


def test_distance_segment_to_segment4():
    f1 = (0, 0)
    f2 = (278.05674689789083, 848.3102386968303)
    t1 = (221.055090540802, 675.7367042826397)
    t2 = (237.6344733521503, 723.4080418578025)
    d, pf, pt, u_f, u_t = de.distance_segment_to_segment(f1, f2, t1, t2)
    if directory:
        plot_distance_segment_to_segment_euc(f1, f2, t1, t2, pf, pt, "test_distance_segment_to_segment4")
    assert d == pytest.approx(0)
    assert pf == pytest.approx((228.63358669727376, 697.5274459946864))
    assert pt == pytest.approx((228.63358669727376, 697.5274459946864))
    assert u_f == pytest.approx(0.8222551304652699)
    assert u_t == pytest.approx(0.4571036354431931)


def test_distance_point_to_segment1():
    locs = [
        (47.6373, -122.0950167),
        (47.6369, -122.0950167),
        (47.6369, -122.0959167),
        (47.6369, -122.09422),
        (47.6369, -122.09400),
        (47.6375, -122.09505)
    ]
    loc_a = (47.6372498273849, -122.094900012016)
    loc_b = (47.6368394494057, -122.094280421734)
    segments = []
    for lat_a, lat_b in itertools.product((loc_a[0], loc_b[0]), repeat=2):
        for lon_a, lon_b in itertools.product((loc_a[1], loc_b[1]), repeat=2):
            segments.append(((lat_a, lon_a), (lat_b, lon_b)))
    # segments = [(loc_a, loc_b)]

    for constrain in [True, False]:
        for loc_idx, loc in enumerate(locs):
            for seg_idx, (loc_a, loc_b) in enumerate(segments):
                dist1, pi1, ti1 = dll.distance_point_to_segment(loc, loc_a, loc_b, constrain=constrain)
                dist2, pi2, ti2 = dll.distance_point_to_segment(loc, loc_b, loc_a, constrain=constrain)
                if directory:
                    plot_distance_point_to_segment_latlon(loc, loc_a, loc_b, pi1,
                                                          f"point_to_segment_{loc_idx}_{seg_idx}_{constrain}.png")
                assert dist1 == pytest.approx(dist2), \
                    f"Locs[{loc_idx},{seg_idx},{constrain}]: Distances different, {dist1} != {dist2}"
                assert pi1[0] == pytest.approx(pi2[0]), \
                    f"Locs[{loc_idx},{seg_idx},{constrain}]: y coord different, {pi1[0]} != {pi2[0]}"
                assert pi1[1] == pytest.approx(pi2[1]), \
                    f"Locs[{loc_idx},{seg_idx},{constrain}]: y coord different, {pi1[1]} != {pi2[1]}"


def plot_distance_point_to_segment_latlon(f, t1, t2, pt, fn):
    import smopy
    import matplotlib.pyplot as plt
    lat_min = min(f[0], t1[0], t2[0])
    lat_max = max(f[0], t1[0], t2[0])
    lon_min = min(f[1], t1[1], t2[1])
    lon_max = max(f[1], t1[1], t2[1])
    bb = [lat_min, lon_min, lat_max, lon_max]
    m = smopy.Map(bb)
    ax = m.show_mpl(figsize=(10, 10))
    p1 = m.to_pixels(t1)
    p2 = m.to_pixels(t2)
    p3 = m.to_pixels(f)
    p4 = m.to_pixels(pt)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o-', color="black")
    ax.plot([p3[0]], [p3[1]], 'o-', color="black")
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], '--', color="red")
    plt.savefig(str(directory / fn))
    plt.close(plt.gcf())


def plot_distance_segment_to_segment_latlon(f1, f2, t1, t2, pf, pt, fn):
    import smopy
    import matplotlib.pyplot as plt
    lat_min = min(f1[0], f2[0], t1[0], t2[0])
    lat_max = max(f1[0], f2[0], t1[0], t2[0])
    lon_min = min(f1[1], f2[1], t1[1], t2[1])
    lon_max = max(f1[1], f2[1], t1[1], t2[1])
    bb = [lat_min, lon_min, lat_max, lon_max]
    m = smopy.Map(bb)
    ax = m.show_mpl(figsize=(10, 10))
    p1 = m.to_pixels(f1)
    p2 = m.to_pixels(f2)
    p3 = m.to_pixels(t1)
    p4 = m.to_pixels(t2)
    p5 = m.to_pixels(pf)
    p6 = m.to_pixels(pt)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'o-')
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], 'o-')
    ax.plot([p5[0], p6[0]], [p5[1], p6[1]], 'x-')
    plt.savefig(str(directory / fn))
    plt.close(plt.gcf())


def plot_distance_segment_to_segment_euc(f1, f2, t1, t2, pf, pt, fn):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot([f1[1], f2[1]], [f1[0], f2[0]], 'o-')
    ax.plot([t1[1], t2[1]], [t1[0], t2[0]], 'o-')
    ax.plot([pf[1], pt[1]], [pf[0], pt[0]], 'x-')
    ax.axis('equal')
    ax.set_aspect('equal')
    plt.savefig(str(directory / fn))
    plt.close(plt.gcf())


if __name__ == "__main__":
    # mm.matching.logger.setLevel(logging.INFO)
    mm.logger.setLevel(logging.DEBUG)
    mm.logger.addHandler(logging.StreamHandler(sys.stdout))
    directory = Path(os.environ.get('TESTDIR', Path(__file__).parent))
    print(f"Saving files to {directory}")
    # test_path_to_gpx()
    # test_grs80()
    # test_distance1()
    # test_bearing1()
    # test_destination1()
    # test_distance_segment_to_segment1()
    # test_distance_segment_to_segment2()
    test_distance_segment_to_segment3()
    # test_distance_segment_to_segment4()
    # test_distance_point_to_segment1()
