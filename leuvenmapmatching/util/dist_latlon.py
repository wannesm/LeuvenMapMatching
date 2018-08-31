# encoding: utf-8
"""
leuvenmapmatching.util.dist_latlon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on:
https://www.movable-type.co.uk/scripts/latlong.html
https://www.movable-type.co.uk/scripts/latlong-vectors.html

:author: Wannes Meert
:copyright: Copyright 2015-2018 DTAI, KU Leuven and Sirris.
:license: Apache License, Version 2.0, see LICENSE for details.
"""
import logging
from math import radians, cos, sin, asin, acos, sqrt, atan2, fabs, degrees, ceil

from . import dist_euclidean as diste


logger = logging.getLogger("be.kuleuven.cs.dtai.mapmatching")
earth_radius = 6371000


def distance(p1, p2):
    """Distance between two points.

    :param p1: (Lat, Lon)
    :param p2: (Lat, Lon)
    :return: Distance in meters
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)
    dist = distance_haversine_radians(lat1, lon1, lat2, lon2)
    return dist


def distance_point_to_segment(p, s1, s2, delta=0.0):
    """Distance between point and segment.

    Cross-track distance.

    https://www.movable-type.co.uk/scripts/latlong.html#cross-track

    :param s1: Segment start point
    :param s2: Segment end point
    :param p: Point to measure distance from path to
    :param delta: Stay away from the endpoints with this factor
    :return: (Distance in meters, projected location on segment, relative location on segment)
    """
    lat1, lon1 = s1
    lat2, lon2 = s2
    lat3, lon3 = p
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)
    lat3, lon3 = radians(lat3), radians(lon3)

    d13 = distance_haversine_radians(lat1, lon1, lat3, lon3, radius=1)
    b13 = bearing_radians(lat1, lon1, lat3, lon3)
    b12 = bearing_radians(lat1, lon1, lat2, lon2)

    dxt = asin(sin(d13) * sin(b13 - b12))
    dist_ct = fabs(dxt) * earth_radius
    dat = acos(cos(d13) / cos(dxt)) * earth_radius
    dist_hs = distance_haversine_radians(lat1, lon1, lat2, lon2)
    ti = dat / dist_hs
    if ti > 1.0:
        ti = 1.0
        lati, loni = lat2, lon2
        dist_ct = distance_haversine_radians(lat3, lon3, lati, loni)
    elif ti < 0.0:
        ti = 0.0
        lati, loni = lat1, lon1
        dist_ct = distance_haversine_radians(lat3, lon3, lati, loni)
    else:
        lati, loni = destination_radians(lat1, lon1, b12, dat)
    pi = (degrees(lati), degrees(loni))

    return dist_ct, pi, ti


def distance_segment_to_segment(f1, f2, t1, t2):
    """Distance between segments. If no intersection within range, simplified to distance from f2 to [t1,t2].

    :param f1: From
    :param f2:
    :param t1: To
    :param t2:
    :return: (distance, proj on f, proj on t, rel pos on t)
    """
    # Translate lat-lon to x-y and apply the Euclidean function
    latf1, lonf1 = f1
    latf1, lonf1 = radians(latf1), radians(lonf1)
    f1 = 0, 0  # Origin

    latf2, lonf2 = f2
    latf2, lonf2 = radians(latf2), radians(lonf2)
    df1f2 = distance_haversine_radians(latf1, lonf1, latf2, lonf2)
    bf1f2 = bearing_radians(latf1, lonf1, latf2, lonf2)
    # print(f"bf1f2 = {bf1f2} = {degrees(bf1f2)} degrees")
    f2 = (df1f2 * cos(bf1f2),  df1f2 * sin(bf1f2))

    latt1, lont1 = t1
    latt1, lont1 = radians(latt1), radians(lont1)
    df1t1 = distance_haversine_radians(latf1, lonf1, latt1, lont1)
    bf1t1 = bearing_radians(latf1, lonf1, latt1, lont1)
    # print(f"bf1t1 = {bf1t1} = {degrees(bf1t1)} degrees")
    t1 = (df1t1 * cos(bf1t1), df1t1 * sin(bf1t1))

    latt2, lont2 = t2
    latt2, lont2 = radians(latt2), radians(lont2)
    dt1t2 = distance_haversine_radians(latt1, lont1, latt2, lont2)
    # print(f"dt1t2 = {dt1t2}")
    bt1t2 = bearing_radians(latt1, lont1, latt2, lont2)
    # print(f"bt1t2 = {bt1t2} = {degrees(bt1t2)} degrees")
    t2 = (t1[0] + dt1t2 * cos(bt1t2), t1[1] + dt1t2 * sin(bt1t2))

    d, pf, pt, u_f, u_t = diste.distance_segment_to_segment(f1, f2, t1, t2)
    pf = destination_radians(latf1, lonf1, bf1f2, u_f * df1f2)
    pf = (degrees(pf[0]), degrees(pf[1]))
    pt = destination_radians(latt1, lont1, bt1t2, u_t * dt1t2)
    pt = (degrees(pt[0]), degrees(pt[1]))

    return d, pf, pt, u_f, u_t


def project(s1, s2, p, delta=0.0):
    _, pi, ti = distance_point_to_segment(p, s1, s2, delta)
    return pi, ti


def box_around_point(p, dist):
    lat, lon = p
    latr, lonr = radians(lat), radians(lon)
    diag_dist = sqrt(2 * dist ** 2)
    lat_t, lon_r = destination_radians(latr, lonr, radians(45), diag_dist)
    lat_b, lon_l = destination_radians(latr, lonr, radians(225), diag_dist)
    lat_t, lon_r = degrees(lat_t), degrees(lon_r)
    lat_b, lon_l = degrees(lat_b), degrees(lon_l)
    return lat_b, lon_l, lat_t, lon_r


def interpolate_path(path, dd):
    """
    :param path: (lat, lon)
    :param dd: Distance difference (meter)
    :return:
    """
    path_new = [path[0]]
    for p1, p2 in zip(path, path[1:]):
        lat1, lon1 = p1[0], p1[1]
        lat2, lon2 = p2[0], p2[1]
        lat1, lon1 = radians(lat1), radians(lon1)
        lat2, lon2 = radians(lat2), radians(lon2)
        dist = distance_haversine_radians(lat1, lon1, lat2, lon2)
        if dist > dd:
            dt = int(ceil(dist / dd))
            distd = dist/dt
            disti = 0
            brng = bearing_radians(lat1, lon1, lat2, lon2)
            for _ in range(dt):
                disti += distd
                lati, loni = destination_radians(lat1, lon1, brng, disti)
                path_new.append((degrees(lati), degrees(loni)))
        path_new.append(p2)
    return path_new


def bearing_radians(lat1, lon1, lat2, lon2):
    """Initial bearing"""
    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return atan2(y, x)


def distance_haversine_radians(lat1, lon1, lat2, lon2, radius=earth_radius):
    lat = lat2 - lat1
    lon = lon2 - lon1
    a = sin(lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lon / 2) ** 2
    dist = 2 * radius * asin(sqrt(a))
    # dist = 2 * radius * atan2(sqrt(a), sqrt(1 - a))
    return dist


def destination_radians(lat1, lon1, bearing, dist):
    d = dist / earth_radius
    lat2 = asin(sin(lat1) * cos(d) + cos(lat1) * sin(d) * cos(bearing))
    lon2 = lon1 + atan2(sin(bearing) * sin(d) * cos(lat1), cos(d) - sin(lat1) * sin(lat2))
    return lat2, lon2


def lines_parallel(f1, f2, t1, t2, d=None):
    latf1, lonf1 = f1
    latf1, lonf1 = radians(latf1), radians(lonf1)
    f1 = 0, 0  # Origin

    latf2, lonf2 = f2
    latf2, lonf2 = radians(latf2), radians(lonf2)
    df1f2 = distance_haversine_radians(latf1, lonf1, latf2, lonf2)
    bf1f2 = bearing_radians(latf1, lonf1, latf2, lonf2)
    # print(f"bf1f2 = {bf1f2} = {degrees(bf1f2)} degrees")
    f2 = (df1f2 * cos(bf1f2), df1f2 * sin(bf1f2))

    latt1, lont1 = t1
    latt1, lont1 = radians(latt1), radians(lont1)
    df1t1 = distance_haversine_radians(latf1, lonf1, latt1, lont1)
    bf1t1 = bearing_radians(latf1, lonf1, latt1, lont1)
    # print(f"bf1t1 = {bf1t1} = {degrees(bf1t1)} degrees")
    t1 = (df1t1 * cos(bf1t1), df1t1 * sin(bf1t1))

    latt2, lont2 = t2
    latt2, lont2 = radians(latt2), radians(lont2)
    dt1t2 = distance_haversine_radians(latt1, lont1, latt2, lont2)
    # print(f"dt1t2 = {dt1t2}")
    bt1t2 = bearing_radians(latt1, lont1, latt2, lont2)
    # print(f"bt1t2 = {bt1t2} = {degrees(bt1t2)} degrees")
    t2 = (t1[0] + dt1t2 * cos(bt1t2), t1[1] + dt1t2 * sin(bt1t2))

    return diste.lines_parallel(f1, f2, t1, t2, d=d)
