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


def latlon2equirectangular(lat, lon, phi_er=0, lambda_er=0):
    """Naive equirectangular projection.  This is the same as considering (lat,lon) == (y,x).
    This is a lot faster but only works if you are far enough from the poles and the dateline.

    :param lat:
    :param lon:
    :param phi_er: The standard parallels (north and south of the equator) where the scale of the projection is true
    :param lambda_er: The central meridian of the map
    """
    x = (lon - lambda_er) * math.cos(phi_er)
    y = lat - phi_er
    return y, x


def equirectangular2latlon(y, x, phi_er=0, lambda_er=0):
    """Naive equirectangular projection. This is the same as considering (lat,lon) == (y,x).
    This is a lot faster but only works if you are far enough from the poles and the dateline.

    :param phi_er: The standard parallels (north and south of the equator) where the scale of the projection is true
    :param lambda_er: The central meridian of the map
    """
    lon = x / math.cos(phi_er) + lambda_er
    lat = y + phi_er
    return lat, lon


def latlon2grs80(coordinates, lon_0=0.0, lat_ts=0.0, y_0=0, x_0=0.0, **kwargs):
    """Given a list of (lon, lat) coordinates, create x-y coordinates in meter.

    :param coordinates: A list of lon-lat tuples
    :param lon_0: Longitude of projection center.
    :param lat_ts: Latitude of true scale. Defines the latitude where scale is not distorted.
    :param y_0: False northing
    :param x_0: False easting
    """
    other_options = " ".join(f"+{key}={val}" for key, val in kwargs.items())
    proj = pyproj.Proj(f"+proj=utm +ellps=GRS80 +units=m "
                       f"+lon_0={lon_0} +lat_ts={lat_ts} +y_0={y_0} +x_0={x_0} "
                       f"+no_defs {other_options}")
    for lon, lat in coordinates:
        x, y = proj(lon, lat)
        yield x, y
