#!/usr/bin/env python3
# encoding: utf-8

import os, sys, logging
from datetime import datetime

try:
    import leuvenmapmatching as mm
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    import leuvenmapmatching as mm


def test_path_to_gpx():
    path = [(i, i, datetime.fromtimestamp(i)) for i in range(0, 10)]
    gpx = mm.util.path_to_gpx(path)

    assert len(path) == len(gpx.tracks[0].segments[0].points)
    assert path[0][0] == gpx.tracks[0].segments[0].points[0].latitude
    assert path[0][1] == gpx.tracks[0].segments[0].points[0].longitude
    assert path[0][2] == gpx.tracks[0].segments[0].points[0].time


if __name__ == "__main__":
    mm.matching.logger.setLevel(logging.INFO)
    # mm.matching.logger.setLevel(logging.DEBUG)
    mm.matching.logger.addHandler(logging.StreamHandler(sys.stdout))
    test_path_to_gpx()