#!/usr/bin/env python3
# encoding: utf-8
import sys
import logging
from datetime import datetime
import pytest

import leuvenmapmatching as mm
from leuvenmapmatching.util.gpx import path_to_gpx
from leuvenmapmatching.util.projections import latlon2grs80


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


if __name__ == "__main__":
    # mm.matching.logger.setLevel(logging.INFO)
    mm.matching.logger.setLevel(logging.DEBUG)
    mm.matching.logger.addHandler(logging.StreamHandler(sys.stdout))
    # test_path_to_gpx()
    test_grs80()
