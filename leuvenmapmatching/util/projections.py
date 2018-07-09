# encoding: utf-8
"""
leuvenmapmatching.util.projections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import math
import logging

import pyproj

logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")
phi_er = 0
lambda_er = 0


def latlon2equirectangular(lat, lon):
    """Naive equirectangular projection.  This is the same as considering (lat,lon) == (y,x).
    This is a lot faster but only works if you are far enough from the poles and the dateline.
    """
    x = (lon - lambda_er) * math.cos(phi_er)
    y = lat - phi_er
    return y, x


def equirectangular2latlon(y, x):
    """Naive equirectangular projection. This is the same as considering (lat,lon) == (y,x).
    This is a lot faster but only works if you are far enough from the poles and the dateline.
    """
    lon = x / math.cos(phi_er) + lambda_er
    lat = y + phi_er
    return lat, lon


def latlon2grs80(coordinates):
    """Given a list of (lon, lat) coordinates, create x-y coordinates in meter."""
    lon_0, lat_0 = coordinates[0]
    proj = pyproj.Proj(f"+proj=merc +ellps=GRS80 +units=m +lon_0={lon_0} +lat_0={lat_0} +lat_ts={lat_0} +no_defs")
    xs, ys = [], []
    for lon, lat in coordinates:
        x, y = proj(lon, lat)
        xs.append(x)
        ys.append(y)
    return zip(xs, ys)
